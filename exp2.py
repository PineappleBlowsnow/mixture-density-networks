import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from Models_training import MDN, train_mdn_early_stop
from config import set_seed
from data import generate_inverse_problem_data
from data_plot import plot_density_compare, plot_mixture_weights, plot_mdn_samples_only, FIGS_DIR, COLORS
from utils import param_count


def plot_test_loss_vs_k(ks, losses, save_path=''):
    plt.figure()
    plt.plot(ks, losses, marker="o", color=COLORS["MDN"])
    plt.xlabel("$K$ (number of components)")
    plt.ylabel("Best test NLL")
    plt.title("Early-stopped test NLL vs $K$")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()



def plot_true_data(data, save_path=''):
    plt.figure()
    plt.scatter(data["x_train"].numpy(), data["y_train"].numpy(), s=16, alpha=0.8, label="true data", color=COLORS["Observations"])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Heteroscedastic data samples")
    plt.legend()
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


def main():
    exp_figs_dir = os.path.join(FIGS_DIR, "exp_2")
    os.makedirs(exp_figs_dir, exist_ok=True)

    set_seed(0)

    os.makedirs("checkpoints", exist_ok=True)

    data = generate_inverse_problem_data(hetero=True)
    plot_true_data(data, save_path=os.path.join(exp_figs_dir, "exp_2_true_hetero_data.pdf"))

    ks = [1, 2, 3, 4, 5, 6, 10]
    best_test_losses = []

    train_loader = DataLoader(
        TensorDataset(data["x_train"], data["y_train"]), batch_size=128, shuffle=True
    )
    test_tensors = (data["x_test"], data["y_test"])

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

        ckpt_path = os.path.join("checkpoints", f"exp2_mdnK{k}.pt")
        torch.save(mdn.state_dict(), ckpt_path)
        print(f"[K={k}] saved checkpoint to {ckpt_path}")

        plot_density_compare(
            mdn,
            xs_to_plot=[0.2, 0.5, 0.8],
            title=f"MDN density $(K={k})$",
            save_path=os.path.join(exp_figs_dir, f"exp_2_mdn_conditional_density_K_{k}.pdf"),
        )
        plot_mixture_weights(
            mdn,
            title=rf"Mixture Weights $\pi_k(x;\theta) \ (K={k})$",
            save_path=os.path.join(exp_figs_dir, f"exp_2_mdn_mixture_weights_K_{k}.pdf"),
        )
        plot_mdn_samples_only(
            mdn,
            x_min=0.0,
            x_max=1.0,
            n_x=200,
            n_samples_per_x=4,
            title=f"MDN samples $(K={k})$",
            save_path=os.path.join(exp_figs_dir, f"exp_2_mdn_samples_only_K_{k}.pdf"),
        )

    plot_test_loss_vs_k(ks, best_test_losses, save_path=os.path.join(exp_figs_dir, "exp_2_mdn_test_nll_vs_k.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
