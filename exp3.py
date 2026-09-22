import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os

from Models_training import MDN, train_mdn_early_stop
from config import set_seed
from data import generate_multi_branch_data, inject_outliers
from data_plot import plot_density_compare, plot_mixture_weights, plot_mdn_samples_only, FIGS_DIR, COLORS
from utils import param_count


def plot_true_with_outliers(data_noisy, outlier_mask, save_path=""):
    x = data_noisy["x_train"].cpu().numpy().squeeze()
    y = data_noisy["y_train"].cpu().numpy().squeeze()

    plt.figure()
    plt.scatter(x, y, s=16, alpha=0.8, label="train (noisy)", color=COLORS["Observations"])
    if outlier_mask is not None:
        idx = outlier_mask.cpu().numpy().astype(bool)
        plt.scatter(x[idx], y[idx], s=28, c="red", marker="x", label="outliers")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Multi-branch data with outliers")
    plt.legend()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def plot_test_loss_vs_k(ks, losses, save_path=""):
    plt.figure()
    plt.plot(ks, losses, marker="o")
    plt.xlabel(r"$K$ (number of components)")
    plt.ylabel("Best test NLL")
    plt.title("Early-stopped test NLL vs $K$")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def main():
    exp_figs_dir = os.path.join(FIGS_DIR, "exp_3")
    os.makedirs(exp_figs_dir, exist_ok=True)
    set_seed(0)

    # 1) Data with outliers
    data_clean = generate_multi_branch_data()
    data_noisy, outlier_mask = inject_outliers(data_clean, frac=0.005, low=-1.0, high=1.5)
    plot_true_with_outliers(data_noisy, outlier_mask, save_path=os.path.join(exp_figs_dir, "exp_3_multi_branch_with_outliers.pdf"))

    # loaders
    train_loader = DataLoader(
        TensorDataset(data_noisy["x_train"], data_noisy["y_train"]), batch_size=256, shuffle=True
    )
    test_tensors = (data_clean["x_test"], data_clean["y_test"])  # evaluate on clean test set

    ks = [3, 5, 7, 10]
    best_test_losses = []

    for k in ks:
        mdn = MDN(in_dim=1, hidden_dim=64, n_components=k, n_hidden_layers=2)
        print(f"K={k}: params={param_count(mdn)}")

        mdn, best_loss = train_mdn_early_stop(
            mdn,
            train_loader=train_loader,
            test_tensors=test_tensors,
            max_epochs=2000,
            lr=1e-3,
            log_every=200,
            patience=300,
        )
        best_test_losses.append(best_loss)

        # Densities at selected x
        plot_density_compare(
            mdn,
            xs_to_plot=[0.2, 0.5, 0.8],
            y_min=-1.0,
            y_max=1.5,
            num_points=400,
            title=f"MDN density $(K={k})$",
            save_path=os.path.join(exp_figs_dir, f"exp_3_multi_branch_mdn_conditional_density_K_{k}.pdf"),
        )

        # Mixture weights
        plot_mixture_weights(
            mdn,
            x_min=0.0,
            x_max=1.0,
            num_points=200,
            title=rf"Mixture Weights $\pi_k(x;\theta) \ (K={k})$",
            save_path=os.path.join(exp_figs_dir, f"exp_3_multi_branch_mdn_mixture_weights_K_{k}.pdf"),
        )

        # MDN samples only
        plot_mdn_samples_only(
            mdn,
            x_min=0.0,
            x_max=1.0,
            n_x=200,
            n_samples_per_x=4,
            title=f"MDN samples $(K={k})$",
            save_path=os.path.join(exp_figs_dir, f"exp_3_multi_branch_mdn_samples_only_K_{k}.pdf"),
        )

    plot_test_loss_vs_k(ks, best_test_losses, save_path=os.path.join(exp_figs_dir, "exp_3_multi_branch_mdn_test_nll_vs_K.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
