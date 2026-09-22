import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

from matplotlib.backends.backend_svg import FigureManagerSVG

from config import device
from utils import gaussian_pdf, mdn_sample

# checkpoint options
SAVE_CHECKPOINTS = False
LOAD_CHECKPOINTS = True
CKPT_DIR = "checkpoints"

FIGS_DIR = "figures"
os.makedirs(FIGS_DIR, exist_ok=True)


# Set up the plotting style
plt.style.use('seaborn-v0_8-paper')

# Configure matplotlib for LaTeX compatibility and high-quality output
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 8),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'axes.axisbelow': True,
    'grid.color': 'gray',
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
})

COLORS = {
    'Observations': '#2ecc71',  # Green
    'MDN': '#3498db',  # Blue
    'MLP': '#e74c3c' # Red
}

LINE_STYLES = ['-', '--', '-.', ':', (0, (5, 10))]

def plot_density_compare(mdn_model, xs_to_plot=None, y_min: float = 0.0, y_max: float = 1.0, num_points: int = 400,
                         title: str = "MDN p(y|x)", save_path: str = ""):
    """Plot MDN p(y|x) for selected x values"""
    mdn_model.eval()
    if xs_to_plot is None:
        xs_to_plot = [0.2, 0.5, 0.8]

    def mdn_density_for_x(mdn, x0, y_vals):
        x_t = torch.tensor([[x0]], dtype=torch.float32, device=device)
        y_grid_t = torch.tensor(y_vals, dtype=torch.float32, device=device).view(-1, 1)
        with torch.no_grad():
            pi, mu, sigma = mdn(x_t)
        K = pi.shape[1]
        Y = y_grid_t.expand(-1, K)
        MU = mu.view(1, -1).expand_as(Y)
        SIG = sigma.view(1, -1).expand_as(Y)
        pdf_comp = gaussian_pdf(Y, MU, SIG)
        pdf = torch.sum(pi.view(1, -1) * pdf_comp, dim=1)
        return pdf.cpu().numpy()

    y_vals = np.linspace(y_min, y_max, num_points)
    plt.figure()
    for i, x0 in enumerate(xs_to_plot):
        pdf_mdn = mdn_density_for_x(mdn_model, x0, y_vals)
        plt.plot(y_vals, pdf_mdn, label=rf"MDN $p_\theta(y \mid x={x0})$", linestyle=LINE_STYLES[i % len(LINE_STYLES)])
    plt.xlabel("$y$")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_mixture_weights(mdn_model, x_min: float = 0.0, x_max: float = 1.0, num_points: int = 200,
                         title: str = "Mixture weights pi_k(x) for p(y|x)", save_path: str = ""):
    """Plot MDN mixture weights pi_k(x)."""
    mdn_model.eval()
    x_grid = torch.linspace(x_min, x_max, num_points).unsqueeze(1).to(device)
    with torch.no_grad():
        pi_x, _, _ = mdn_model(x_grid)

    x_np = x_grid.cpu().numpy().squeeze()
    pi_np = pi_x.cpu().numpy()

    plt.figure()
    for k in range(pi_np.shape[1]):
        plt.plot(x_np, pi_np[:, k], label=rf"$\pi_{k}(x;\theta)$", linestyle=LINE_STYLES[k % len(LINE_STYLES)])
    plt.xlabel("$x$")
    plt.ylabel(r"mixture weight $\pi_k(x;\theta)$")
    plt.title(title)
    plt.legend()

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_mdn_samples_only(mdn_model: torch.nn.Module, x_min: float = 0.0, x_max: float = 1.0, n_x: int = 200,
                          n_samples_per_x: int = 4, title: str = "MDN samples", save_path: str = ""):
    """Scatter only MDN samples p(y|x) on a uniform x grid."""
    mdn_model.eval()
    x_vals = np.linspace(x_min, x_max, n_x)
    x_samp_list, y_samp_list = [], []
    for xv in x_vals:
        x_t = torch.tensor([[xv]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pi_x, mu_x, sigma_x = mdn_model(x_t)
            y_samples = mdn_sample(pi_x, mu_x, sigma_x, n_samples=n_samples_per_x)
        y_np = y_samples.squeeze(0).cpu().numpy()
        x_jitter = xv + 0.002 * np.random.randn(y_np.shape[0])
        x_samp_list.append(x_jitter)
        y_samp_list.append(y_np)

    x_samp_all = np.concatenate(x_samp_list)
    y_samp_all = np.concatenate(y_samp_list)

    plt.figure()
    plt.scatter(x_samp_all, y_samp_all, label="MDN samples", color=COLORS["MDN"], s=16, alpha=0.8)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title(title)
    plt.legend()

    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

