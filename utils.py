import math
import os

import torch

from config import device

SAVE_CHECKPOINTS = True
LOAD_CHECKPOINTS = True
CKPT_DIR = "checkpoints"


def param_count(model: torch.nn.Module) -> int:
    """Number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load(model, path):
    if LOAD_CHECKPOINTS and os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded checkpoint: {path}")
        return True
    return False


def save(model, path):
    if SAVE_CHECKPOINTS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        print(f"Saved checkpoint: {path}")


def mdn_gaussian_nll(y: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Define Negative log-likelihood as loss function for 1D Gaussian mixture outputs."""
    y_exp = y.expand_as(mu)
    var = sigma ** 2

    # Compute log probabilities to avoid numerical issues
    log_prob = -0.5 * (
            math.log(2 * math.pi) + torch.log(var) + (y_exp - mu) ** 2 / var
    )

    log_pi = torch.log(pi + 1e-8)  # Avoid log(0)
    log_weighted = log_pi + log_prob  # log (pi_k * N(y|mu_k, sigma_k))

    # LOG SUM EXP TRICK
    log_sum_exp = torch.logsumexp(log_weighted, dim=1, keepdim=True)

    return -torch.mean(log_sum_exp)


def eval_mdn_nll(model, x_test, y_test):
    """Evaluate MDN on test set and return NLL score."""
    model.eval()
    x_dev = x_test.to(device)
    y_dev = y_test.to(device)
    with torch.no_grad():
        pi, mu, sigma = model(x_dev)
        return float(mdn_gaussian_nll(y_dev, pi, mu, sigma).item())


def mdn_sample(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """Sample from MDN output for a batch of inputs."""
    B, K = pi.shape
    # step 1: select component based on pi
    cat = torch.distributions.Categorical(pi)
    comp_idx = cat.sample((n_samples,)).T  # (B, n_samples)

    mu_exp = mu.unsqueeze(1).expand(B, n_samples, K)
    sigma_exp = sigma.unsqueeze(1).expand(B, n_samples, K)

    idx = comp_idx.unsqueeze(-1)
    chosen_mu = torch.gather(mu_exp, 2, idx).squeeze(-1)
    chosen_sigma = torch.gather(sigma_exp, 2, idx).squeeze(-1)

    eps = torch.randn_like(chosen_mu)
    return chosen_mu + chosen_sigma * eps


def gaussian_pdf(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Compute univariate Gaussian PDF value for tensors (element-wise)."""
    var = (sigma ** 2).clamp(min=1e-12)
    coef = 1.0 / torch.sqrt(2 * math.pi * var)
    return coef * torch.exp(-0.5 * (x - mu) ** 2 / var)


__all__ = [
    "mdn_gaussian_nll",
    "mdn_sample",
    "gaussian_pdf",
]
