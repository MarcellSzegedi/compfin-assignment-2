"""Plots the difference between control variate and Heston model asian option estimation."""

import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from compfin_assignment_2.asian_option.heston_model.option_pricing import (
    diff_control_vs_heston_stat,
)

app = typer.Typer()


@app.command(name="control-vs-heston-stat")
def control_vs_heston_stat():
    """Plots the difference between control variate and Heston model asian option estimation."""
    settings_example = {
        "n_trajectories": 1000,
        "s_0": 100,
        "v_0": 0.2**2,
        "t_end": 1,
        "drift": 0.02,
        "kappa": 6,
        "vol_of_vol": 0.15,
        "stoc_inc_corr": -0.7,
        "num_steps": 1000,
        "risk_free_rate": 0.02,
        "alpha": 0.05,
    }
    strike_prices = np.linspace(10, 150, 1000)

    control_est = []
    heston_est = []
    control_stde = []
    heston_stde = []
    control_var = []
    heston_var = []

    for strike in tqdm(strike_prices, desc="Strike Prices"):
        curr_config = settings_example.copy()
        curr_config["strike"] = strike
        (cont_est, cont_var, cont_std, hest_est, hest_var, hest_std) = diff_control_vs_heston_stat(
            curr_config
        )

        control_est.append(cont_est)
        heston_est.append(hest_est)
        control_stde.append(cont_std)
        heston_stde.append(hest_std)
        control_var.append(cont_var)
        heston_var.append(hest_var)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6), constrained_layout=True)
    axes[0].plot(strike_prices, control_est, label="Control Variate")
    axes[0].plot(strike_prices, heston_est, label="Heston Model")
    axes[0].set_xlabel("Strike Price")
    axes[0].set_ylabel("Asian Call Option Price")
    axes[0].legend()
    axes[0].set_title("Arithmetic Asian call Option Price Estimation")

    axes[1].plot(strike_prices, control_stde, label="Control Variate")
    axes[1].plot(strike_prices, heston_stde, label="Heston Model")
    axes[1].set_xlabel("Strike Price")
    axes[1].set_ylabel("Standard Error")
    axes[1].legend()
    axes[1].set_title("Arithmetic Asian call Option Price Estimation Standard Error")

    axes[2].plot(strike_prices, control_var, label="Control Variate")
    axes[2].plot(strike_prices, heston_var, label="Heston Model")
    axes[2].set_xlabel("Strike Price")
    axes[2].set_ylabel("Variance")
    axes[2].legend()
    axes[2].set_title("Arithmetic Asian call Option Price Estimation Variance")

    plt.savefig("results/figures/control_vs_heston_stat.png", dpi=600)
    plt.show()
