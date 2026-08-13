import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from src.simulator import PathData


def plot_representative_path(path: PathData, gamma: float, dt: float, save_path: str) -> None:
    """
    Plot mid-price S_t, reservation price r_t, ask and bid quotes on the same axes.
    Saves figure to save_path.
    """
    N = len(path.r)
    t_full  = np.arange(N + 1) * dt   # length N+1, for S
    t_steps = np.arange(N) * dt        # length N,   for quotes

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_full,  path.S,     color="black",  linewidth=1.0, label="Mid-price $S_t$")
    ax.plot(t_steps, path.r,     color="blue",   linewidth=1.0, linestyle="--",
            label="Reservation price $r_t$")
    ax.plot(t_steps, path.p_ask, color="red",    linewidth=0.8, linestyle="-.",
            label="Ask $p^a_t$")
    ax.plot(t_steps, path.p_bid, color="green",  linewidth=0.8, linestyle="-.",
            label="Bid $p^b_t$")

    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Price")
    ax.set_title(f"Representative path — $\\gamma = {gamma}$")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_profit_histograms(
    profits_inv: np.ndarray,
    profits_sym: np.ndarray,
    gamma: float,
    save_path: str,
) -> None:
    """
    Overlaid histograms of terminal profits for inventory vs symmetric strategies.
    Uses alpha=0.6, 30 bins; includes mean and std in legend labels.
    Saves figure to save_path.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    mu_inv, sd_inv = np.mean(profits_inv), np.std(profits_inv)
    mu_sym, sd_sym = np.mean(profits_sym), np.std(profits_sym)

    ax.hist(
        profits_inv,
        bins=30,
        alpha=0.6,
        color="steelblue",
        label=f"Inventory  μ={mu_inv:.2f}  σ={sd_inv:.2f}",
    )
    ax.hist(
        profits_sym,
        bins=30,
        alpha=0.6,
        color="darkorange",
        label=f"Symmetric  μ={mu_sym:.2f}  σ={sd_sym:.2f}",
    )

    ax.set_xlabel("Terminal profit")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Terminal profit distribution — $\\gamma = {gamma}$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_all_gammas(results: dict, save_dir: str) -> None:
    """
    Produce one profit histogram per gamma scenario.

    results: {gamma: {'inventory': profits_array, 'symmetric': profits_array}}
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for gamma, strategies in results.items():
        profits_inv = strategies["inventory"]
        profits_sym = strategies["symmetric"]
        filename = os.path.join(save_dir, f"profit_hist_gamma_{gamma}.png")
        plot_profit_histograms(profits_inv, profits_sym, gamma, filename)
