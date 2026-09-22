import copy
import os
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import device
from utils import save, load, mdn_gaussian_nll

TensorPair = Tuple[torch.Tensor, torch.Tensor]

# checkpoint options (uses maybe_save/maybe_load from utils)
SAVE_CHECKPOINTS = True
LOAD_CHECKPOINTS = True
CKPT_DIR = "checkpoints"


class MLPRegressor(nn.Module):
    """Simple MLP for 1D regression with two hidden layers of 64 units, ReLU activations."""

    def __init__(self, in_dim: int = 1, hidden_dim: int = 64, out_dim: int = 1, n_hidden_layers: int = 2):
        super().__init__()
        layers = []
        last_dim = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_regressor(model: nn.Module, train_loader: DataLoader, test_tensors: TensorPair, n_epochs: int = 2000,
                        lr: float = 1e-3, log_every: int = 200, ) -> nn.Module:
    """Train simple MLP regressor with MSE loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_test, y_test = (t.to(device) for t in test_tensors)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        if epoch % log_every == 0 or epoch == 1 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                y_pred_test = model(x_test)
                test_mse = criterion(y_pred_test, y_test).item()
            print(f"[MLP] Epoch {epoch:4d} | train MSE: {train_loss:.4f} | test MSE: {test_mse:.4f}")

    return model


class MDN(nn.Module):
    """Mixture Density Network with K Gaussian components for 1D outputs."""

    def __init__(self, in_dim: int = 1, hidden_dim: int = 64, n_components: int = 3, n_hidden_layers: int = 2):
        super().__init__()
        self.K = n_components

        layers = []
        last_dim = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.feature = nn.Sequential(*layers)

        self.pi_head = nn.Linear(hidden_dim,
                                 self.K)  # Linear transform features to mixture weights，shape: [batch_size, K]
        self.mu_head = nn.Linear(hidden_dim, self.K)
        self.sigma_head = nn.Linear(hidden_dim, self.K)

    def forward(self, x: torch.Tensor):
        h = self.feature(x)
        pi_logits = self.pi_head(h)
        pi = F.softmax(pi_logits, dim=-1)  # softmax to get mixture weights, sum to 1
        mu = self.mu_head(h)  # no constraint with mu
        sigma = F.softplus(self.sigma_head(h)) + 1e-6  # softplus to ensure sigma is positive
        return pi, mu, sigma


def train_mdn(model: nn.Module, train_loader: DataLoader, test_tensors: TensorPair, n_epochs: int = 2000,
              lr: float = 1e-3, log_every: int = 200, verbose: bool = True, ) -> nn.Module:
    """Train MDN with NLL loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_test, y_test = (t.to(device) for t in test_tensors)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pi, mu, sigma = model(xb)
            loss = mdn_gaussian_nll(yb, pi, mu, sigma)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_nll = running_loss / len(train_loader.dataset)

        if verbose and (epoch % log_every == 0 or epoch == 1 or epoch == n_epochs):
            model.eval()
            with torch.no_grad():
                pi_test, mu_test, sigma_test = model(x_test)
                test_nll = mdn_gaussian_nll(y_test, pi_test, mu_test, sigma_test).item()
            print(f"[MDN] Epoch {epoch:4d} | train NLL: {train_nll:.4f} | test NLL: {test_nll:.4f}")

    return model


def train_mdn_early_stop(model: nn.Module, train_loader: DataLoader, test_tensors: TensorPair, max_epochs: int = 2000,
                         lr: float = 1e-3, log_every: int = 200, patience: int = 300, ) -> Tuple[nn.Module, float]:
    """Train MDN with early stopping on test NLL; returns best model and best loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_test, y_test = (t.to(device) for t in test_tensors)

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pi, mu, sigma = model(xb)
            loss = mdn_gaussian_nll(yb, pi, mu, sigma)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_nll = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            pi_t, mu_t, sigma_t = model(x_test)
            test_nll = mdn_gaussian_nll(y_test, pi_t, mu_t, sigma_t).item()

        if test_nll < best_loss:
            best_loss = test_nll
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(
                f"[K={model.K}] Epoch {epoch:4d} | train NLL: {train_nll:.4f} | test NLL: {test_nll:.4f} | best: {best_loss:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improve for {patience} epochs).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss


def run_mdn_experiment(data_fn, mdn_components: int, checkpoint_prefix: Optional[str] = None, x_key: str = "x_train",
                       y_key: str = "y_train", batch_size: int = 128, n_epochs: int = 2000, lr: float = 1e-3,
                       log_every: int = 400,
                       hidden_dim: int = 64, n_hidden_layers: int = 2, ):
    """Train MDN experiment helper for 1D."""
    data = data_fn()
    x_train, y_train = data[x_key], data[y_key]
    x_test, y_test = data["x_test"], data["y_test"]

    ds = TensorDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    use_ckpt = checkpoint_prefix is not None
    mdn_ckpt = (os.path.join(CKPT_DIR, f"{checkpoint_prefix}_mdnK{mdn_components}.pt") if use_ckpt else None)

    mdn = MDN(in_dim=1, hidden_dim=hidden_dim, n_components=mdn_components, n_hidden_layers=n_hidden_layers)
    if not (use_ckpt and LOAD_CHECKPOINTS and load(mdn, mdn_ckpt)):
        mdn = train_mdn(mdn, dl, test_tensors=(x_test, y_test), n_epochs=n_epochs, lr=lr, log_every=log_every)
        if use_ckpt and SAVE_CHECKPOINTS:
            save(mdn, mdn_ckpt)

    return {"data": data, "mdn": mdn}


__all__ = ["MLPRegressor", "MDN", "mdn_gaussian_nll", "train_mlp_regressor", "train_mdn", "train_mdn_early_stop",
           "TensorPair", "run_mdn_experiment", ]
