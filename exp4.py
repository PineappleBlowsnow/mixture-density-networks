import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torch.utils.data import DataLoader, TensorDataset

from config import device, set_seed
from data import generate_star_with_circle
from data_plot import FIGS_DIR, COLORS


def gaussian_nll_2d(y, pi, mu, sigma):
    """Diagonal Gaussian mixture NLL for 2D targets.
    y: (B,2); pi: (B,K); mu: (B,K,2); sigma: (B,K,2)
    """
    B, K, _ = mu.shape
    y_exp = y.unsqueeze(1).expand(-1, K, -1)
    var = sigma ** 2
    log_det = torch.log(var.prod(dim=2) + 1e-12)  # log determinant of diagonal covariance
    maha = ((y_exp - mu) ** 2 / (var + 1e-12)).sum(dim=2)  # Mahalanobis distance
    log_comp = -0.5 * (log_det + maha + 2 * np.log(2 * np.pi))
    log_pi = torch.log(pi + 1e-12)
    log_mix = torch.logsumexp(log_pi + log_comp, dim=1)
    return -log_mix.mean()


class MDNFull(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, n_hidden=2, K=8):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            last = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.pi_head = nn.Linear(hidden_dim, K)
        self.mu_head = nn.Linear(hidden_dim, K * 2)
        self.sigma_head = nn.Linear(hidden_dim, K * 2)
        self.K = K

    def forward(self, x):
        h = self.backbone(x)
        pi = F.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.K, 2)
        sigma = F.softplus(self.sigma_head(h)).view(-1, self.K, 2) + 1e-3
        return pi, mu, sigma


class GatedDual(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, n_hidden=2, K=8):
        super().__init__()
        self.K = K
        # shared backbone for gating
        layers = []
        last = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            last = hidden_dim
        self.gating = nn.Sequential(*layers)
        self.gate_head = nn.Linear(hidden_dim, K)
        # expert A (2 components)   
        self.expA = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # mu(2 comps *2 dims) + sigma(2 comps *2 dims)
        )
        # expert B (6 components)
        self.expB = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 24),  # mu(6*2) + sigma(6*2)
        )

    def forward(self, x):
        h = self.gating(x)
        pi = F.softmax(self.gate_head(h), dim=-1)
        outA = self.expA(x)
        muA = outA[:, :4].view(-1, 2, 2)
        sigmaA = F.softplus(outA[:, 4:]).view(-1, 2, 2) + 1e-3

        outB = self.expB(x)
        muB = outB[:, :12].view(-1, 6, 2)
        sigmaB = F.softplus(outB[:, 12:]).view(-1, 6, 2) + 1e-3

        # concatenate mu/sigma to shape (B,8,2)
        mu = torch.cat([muA, muB], dim=1)
        sigma = torch.cat([sigmaA, sigmaB], dim=1)
        return pi, mu, sigma


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, train_loader, test_tensors, max_epochs=2000, lr=1e-3, log_every=100, patience=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_test, y_test = (t.to(device) for t in test_tensors)
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pi, mu, sigma = model(xb)
            loss = gaussian_nll_2d(yb, pi, mu, sigma)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)
        model.eval()
        with torch.no_grad():
            pi_t, mu_t, sigma_t = model(x_test)
            test_loss = gaussian_nll_2d(y_test, pi_t, mu_t, sigma_t).item()
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if log_every and (ep % log_every == 0 or ep == 1):
            print(f"Epoch {ep:4d} | train {train_loss:.4f} | test {test_loss:.4f} | best {best_loss:.4f}")
        if no_improve >= patience:
            print(f"Early stop at epoch {ep} (patience {patience})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def sample_mdn(model, x_in):
    model.eval()
    with torch.no_grad():
        pi, mu, sigma = model(x_in)
        # pi: (B,K), mu/sigma: (B,K,2)
        batch_size, K, out_dim = mu.shape

        # Draw 1 sample from the multinomial distribution defined by pi.

        k_indices = torch.multinomial(pi, num_samples=1).squeeze(1)
        # We need to use 'gather' to pick the specific mu/sigma corresponding to k_indices.
        gather_indices = k_indices.view(-1, 1, 1).expand(-1, 1, out_dim)

        # Extract the specific mean and std dev for the chosen component

        mu_selected = mu.gather(1, gather_indices).squeeze(1)
        sigma_selected = sigma.gather(1, gather_indices).squeeze(1)

        # Generate standard normal noise: epsilon ~ N(0, 1)
        epsilon = torch.randn_like(mu_selected)

        # Apply transformation: y = mu + sigma * epsilon
        y_sampled = mu_selected + (sigma_selected * epsilon)

        return y_sampled


def sample_gated(model, x_in):
    """Inference-time sampling: gate first, then run only the needed expert per sample."""
    model.eval()
    with torch.no_grad():
        # component selection
        h = model.gating(x_in)
        pi = F.softmax(model.gate_head(h), dim=-1)  # (B, K)
        comp = torch.multinomial(pi, num_samples=1).squeeze(1)  # (B,)

        B = x_in.shape[0]
        mu_sel = torch.empty(B, 2, device=x_in.device)
        sigma_sel = torch.empty(B, 2, device=x_in.device)

        idxA = (comp < 2).nonzero(as_tuple=True)[0]  # comps 0,1 -> expA
        idxB = (comp >= 2).nonzero(as_tuple=True)[0]  # comps 2..7 -> expB

        if len(idxA) > 0:
            xa = x_in[idxA]
            outA = model.expA(xa)
            muA = outA[:, :4].view(-1, 2, 2)
            sigmaA = F.softplus(outA[:, 4:]).view(-1, 2, 2) + 1e-3
            ca = comp[idxA]  # 0/1
            mu_sel[idxA] = muA[torch.arange(len(idxA)), ca]
            sigma_sel[idxA] = sigmaA[torch.arange(len(idxA)), ca]

        if len(idxB) > 0:
            xb = x_in[idxB]
            outB = model.expB(xb)
            muB = outB[:, :12].view(-1, 6, 2)
            sigmaB = F.softplus(outB[:, 12:]).view(-1, 6, 2) + 1e-3
            cb = comp[idxB] - 2  # map 2..7 -> 0..5
            mu_sel[idxB] = muB[torch.arange(len(idxB)), cb]
            sigma_sel[idxB] = sigmaB[torch.arange(len(idxB)), cb]

        eps = torch.randn_like(mu_sel)
        return mu_sel + sigma_sel * eps


def plot_data(data, save_path=""):
    arr = data.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(arr[:, 0], arr[:, 1], s=16, alpha=0.8, label="observations", color=COLORS["Observations"])
    plt.axis("equal")
    plt.title("Star + circle data")
    plt.legend()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_samples(samples, title, save_path=""):
    arr = samples.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(arr[:, 0], arr[:, 1], s=16, alpha=0.8, label="samples", color=COLORS["MDN"])
    plt.axis("equal")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()



def compare_flops(mdn_model, gated_model, input_dim=2):
    """
    Compares FLOPs. 
    Now supports AUTO-DETECTION of hidden_dim. No need to pass it manually.
    """

    # 1. Setup Device
    device = next(mdn_model.parameters()).device
    dummy_input = torch.randn(1, input_dim).to(device)

    print("\n" + "=" * 50)
    print("     FLOPs Comparison - MDNFull vs GatedDual")
    print("=" * 50)
    # 1. Profile MDNFull (Baseline)
    flops_mdn, _ = profile(mdn_model, inputs=(dummy_input,), verbose=False)
    print(f"[MDNFull] Full Inference Cost:   {int(flops_mdn):,} FLOPs")

    #
    # 2. Profile GatedDual
    detected_hidden_dim = gated_model.gate_head.in_features

    dummy_hidden = torch.randn(1, detected_hidden_dim).to(device)

    # --- Step A: Common Cost ---
    flops_gating, _ = profile(gated_model.gating, inputs=(dummy_input,), verbose=False)
    flops_gate_head, _ = profile(gated_model.gate_head, inputs=(dummy_hidden,), verbose=False)
    base_cost = flops_gating + flops_gate_head

    # --- Step B: Path Costs ---
    flops_expA, _ = profile(gated_model.expA, inputs=(dummy_input,), verbose=False)
    cost_path_A = base_cost + flops_expA

    flops_expB, _ = profile(gated_model.expB, inputs=(dummy_input,), verbose=False)
    cost_path_B = base_cost + flops_expB

    # 3. Results
    print("-" * 50)
    print(f"[GatedDual] Hidden Dim Detected: {detected_hidden_dim}")  # 打印出来确认一下
    print(f"[GatedDual] Path A (ExpA):       {int(cost_path_A):,} FLOPs")
    print(f"[GatedDual] Path B (ExpB):       {int(cost_path_B):,} FLOPs")

    avg_flops_gated = (cost_path_A + cost_path_B) / 2
    savings = (1 - avg_flops_gated / flops_mdn) * 100

    print("-" * 50)
    print(f"Conclusion:")
    print(f"  Efficiency Gain:    {savings:.2f}% reduction")
    print("=" * 50 + "\n")


def main():
    exp_figs_dir = os.path.join(FIGS_DIR, "exp_4")
    os.makedirs(exp_figs_dir, exist_ok=True)

    set_seed(0)
    os.makedirs("checkpoints", exist_ok=True)

    ds = generate_star_with_circle()
    data = ds["data"].to(device)
    # add uniform noise to star + circle
    data = data + torch.empty_like(data).uniform_(-0.1, 0.1)
    n_total = data.shape[0]
    n_train = int(0.8 * n_total)
    train_subset, test_subset = torch.utils.data.random_split(data, [n_train, n_total - n_train])

    train_coords = train_subset.dataset[train_subset.indices].to(device)
    test_coords = test_subset.dataset[test_subset.indices].to(device)

    # Use coordinates both as conditioning input and target (unconditional density surrogate)
    train_loader = DataLoader(TensorDataset(train_coords, train_coords), batch_size=256, shuffle=True)
    test_tensors = (test_coords, test_coords)

    plot_data(data.cpu(), save_path=os.path.join(exp_figs_dir, "exp_4_star_circle_data.pdf"))

    # Model 1: standard MDN, K=8
    mdn = MDNFull(in_dim=2, hidden_dim=96, n_hidden=3, K=8).to(device)
    print(f"MDNFull params: {param_count(mdn)}")
    mdn, mdn_best = train_model(mdn, train_loader, test_tensors, max_epochs=2000, lr=1e-3, log_every=200, patience=200)
    torch.save(mdn.state_dict(), "checkpoints/exp4_mdnfull.pt")
    # sample
    x_sample = data  # same x used for both sampling processes
    t0 = time.perf_counter()
    mdn_samples = sample_mdn(mdn, x_sample)
    t1 = time.perf_counter()
    plot_samples(mdn_samples.cpu(), "MDNFull samples", save_path=os.path.join(exp_figs_dir, "exp_4_mdnfull_samples.pdf"))
    print(f"MDN sampling time: {t1 - t0:.4f}s for {len(x_sample)} points")

    # Model 2: gating + dual experts
    gated = GatedDual(in_dim=2, hidden_dim=94, n_hidden=3, K=8).to(device)
    print(f"GatedDual params: {param_count(gated)}")
    gated, gated_best = train_model(gated, train_loader, test_tensors, max_epochs=2000, lr=1e-3, log_every=200,
                                    patience=200)
    torch.save(gated.state_dict(), "checkpoints/exp4_gateddual.pt")
    # use the same coordinate distribution for sampling (matches training support)
    x_sample_new = data[torch.randperm(len(data))]
    t2 = time.perf_counter()
    gated_samples = sample_gated(gated, x_sample_new)
    t3 = time.perf_counter()
    plot_samples(gated_samples.cpu(), "GatedDual samples", save_path=os.path.join(exp_figs_dir, "exp_4_gateddual_samples.pdf"))
    print(f"Gated sampling time: {t3 - t2:.4f}s for {len(x_sample_new)} points")
    # Compare FLOPs
    compare_flops(mdn, gated, input_dim=2)

    plt.show()


if __name__ == "__main__":
    main()
