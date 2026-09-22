import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from Models_training import run_mdn_experiment, MLPRegressor, train_mlp_regressor
from config import device, set_seed
from data import generate_inverse_problem_data
from data_plot import plot_density_compare, plot_mixture_weights, FIGS_DIR, COLORS
from utils import mdn_sample


def plot_forward(model: torch.nn.Module, save_path: str = ""):
    # noise-free ground truth y->x curve
    y_clean = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    x_clean = y_clean + 0.3 * np.sin(2 * np.pi * y_clean)

    y_grid = torch.from_numpy(y_clean).float().to(device)
    with torch.no_grad():
        x_pred = model(y_grid).cpu().numpy()

    plt.figure()
    plt.plot(y_clean, x_clean, label="true (noise-free)", color=COLORS["Observations"])
    plt.plot(y_clean, x_pred, label="MLP forward", color=COLORS["MLP"])
    plt.xlabel("$y$")
    plt.ylabel("$x$")
    plt.title(r"Forward mapping $y \mapsto x$ vs MLP")
    plt.legend()

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()



def plot_inverse(model: torch.nn.Module, save_path: str = ""):
    # noise-free ground truth x->y curve (paramized by y)
    y_clean = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    x_clean = y_clean + 0.3 * np.sin(2 * np.pi * y_clean)
    x_clean_t = torch.from_numpy(x_clean).float().to(device)
    with torch.no_grad():
        y_pred = model(x_clean_t).cpu().numpy()

    plt.figure()
    plt.plot(x_clean, y_clean, label="true", color=COLORS["Observations"])
    plt.plot(x_clean, y_pred, label="MLP inverse", color=COLORS["MLP"])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(r"Inverse mapping $x \mapsto y$ vs MLP")
    plt.legend()

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_mdn_samples_vs_true(mdn_model: torch.nn.Module, save_path: str = ""):
    """Sample from MDN at many x and compare to the noise-free inverse curve."""
    mdn_model.eval()
    y_clean = np.linspace(0.0, 1.0, 400).reshape(-1, 1)
    x_clean = y_clean + 0.3 * np.sin(2 * np.pi * y_clean)

    # uniform x grid to avoid oversampling fold points
    x_vals = np.linspace(0.0, 1.0, 200)
    x_samp_list, y_samp_list = [], []
    for xv in x_vals:
        x_t = torch.tensor([[xv]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pi_x, mu_x, sigma_x = mdn_model(x_t)
            y_samples = mdn_sample(pi_x, mu_x, sigma_x, n_samples=3)
        y_np = y_samples.squeeze(0).cpu().numpy()
        x_jitter = xv + 0.002 * np.random.randn(y_np.shape[0])
        x_samp_list.append(x_jitter)
        y_samp_list.append(y_np)

    x_samp_all = np.concatenate(x_samp_list)
    y_samp_all = np.concatenate(y_samp_list)

    plt.figure()
    plt.plot(x_clean, y_clean, label="true", color=COLORS["Observations"])
    plt.scatter(x_samp_all, y_samp_all, s=16, alpha=0.8, label="MDN samples", color=COLORS["MDN"])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(r"Inverse mapping $x \mapsto y$ vs MDN")
    plt.legend()

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def main():
    exp_figs_dir = os.path.join(FIGS_DIR, "exp_1")
    os.makedirs(exp_figs_dir, exist_ok=True)
    set_seed(42)
    # === Forward task: y -> x with MLP ===
    data_forward = generate_inverse_problem_data(hetero=False)
    mlp_forward = train_mlp_regressor(
        MLPRegressor(),
        train_loader=torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_forward["y_train"], data_forward["x_train"]),
            batch_size=128,
            shuffle=True,
        ),
        test_tensors=(data_forward["y_test"], data_forward["x_test"]),
        n_epochs=2000,
        lr=1e-3,
        log_every=400,
    )
    plot_forward(mlp_forward, save_path=os.path.join(exp_figs_dir, "exp_1_forward_mlp.pdf"))

    #  Inverse task: x -> y with MLP (same data, swap roles) 
    mlp_inverse = train_mlp_regressor(
        MLPRegressor(),
        train_loader=torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_forward["x_train"], data_forward["y_train"]),
            batch_size=128,
            shuffle=True,
        ),
        test_tensors=(data_forward["x_test"], data_forward["y_test"]),
        n_epochs=2000,
        lr=1e-3,
        log_every=400,
    )
    plot_inverse(mlp_inverse, save_path=os.path.join(exp_figs_dir, "exp_1_inverse_mlp.pdf"))

    # === MDN on inverse task (K=3) ===
    mdn_results = run_mdn_experiment(
        data_fn=lambda: generate_inverse_problem_data(hetero=False),
        mdn_components=3,
        checkpoint_prefix=None,
        x_key="x_train",
        y_key="y_train",
        n_epochs=2000,
        lr=1e-3,
        log_every=400,
    )

    # Density at a few x
    plot_density_compare(
        mdn_results["mdn"],
        xs_to_plot=[0.2, 0.5, 0.8],
        title=r"MDN density at $x=0.2,0.5,0.8 \ (K=3)$",
        save_path=os.path.join(exp_figs_dir, "exp_1_mdn_conditional_densities.pdf")
    )

    # MDN samples vs true curve
    plot_mdn_samples_vs_true(mdn_results["mdn"], save_path=os.path.join(exp_figs_dir, "exp_1_mdn_samples_vs_true.pdf"))

    # Mixture weights
    plot_mixture_weights(
        mdn_results["mdn"],
        title=r"Mixture Weights $\pi_k(x;\theta) \ (K=3)$",
        save_path=os.path.join(exp_figs_dir, "exp_1_mdn_mixture_weights.pdf")
    )

    plt.show()


if __name__ == "__main__":
    main()
