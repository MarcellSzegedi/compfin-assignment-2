"""Utils for the benchmarks."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

models_settings = {
    "s_0": 100,
    "v_0": 0.2**2,
    "t_end": 1,
    "drift": 0,
    "kappa": 0,
    "theta": 0.2**2,
    "vol_of_vol": 0,
    "stoc_inc_corr": -0.7,
    "risk_free_rate": 0.02,
    "alpha": 0.05,
}


def plot_convergence_order_difference(
    euler_results: dict[str, npt.NDArray[np.float64]],
    milstein_results: dict[str, npt.NDArray[np.float64]],
    n_time_steps: npt.NDArray[np.float64],
    title: str,
    file_name: Optional[str] = None,
) -> None:
    """Plots the order of convergence of the Euler and Milstein schemes."""
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 12), constrained_layout=True)

    for ax_idx, strike_price in enumerate(euler_results.keys()):
        ax = axes[ax_idx // 2, ax_idx % 2]
        ax.plot(1 / n_time_steps, euler_results[strike_price], label="Euler")
        ax.plot(1 / n_time_steps, milstein_results[strike_price], label="Milstein")
        ax.set_title(f"Strike price: {int(strike_price)}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel("Log Time step")
        ax.set_ylabel("Log Error")

    fig.suptitle(title, fontsize=16)

    if file_name is not None:
        plt.savefig(f"results/figures/{file_name}.png", dpi=600)

    plt.show()
