"""Plots the GBM volatility vs control variate variance."""

import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from compfin_assignment_2.asian_option.heston_model.option_pricing import cont_est_var_vs_gbm_vol

app = typer.Typer()


@app.command(name="gbm-vol-vs-control-var")
def gbm_vol_vs_control_var() -> None:
    """Plots the GBM volatility vs control variate variance."""
    stirke_prices = np.array([1, 20, 40, 60, 80, 100, 120, 140])
    gbm_vol = np.linspace(0.001, 2, 1000)
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
        "strike": 100,
    }

    control_variate_var = {}

    for strike_price in tqdm(stirke_prices, desc="Strike Prices", leave=False, position=0):
        control_variate_var[strike_price] = [
            cont_est_var_vs_gbm_vol(settings_example, gbm_vol=gbm_vol_curr)
            for gbm_vol_curr in tqdm(gbm_vol, desc="GBM Volatility", leave=False, position=1)
        ]

    plt.figure(figsize=(10, 6))
    for strike_price in stirke_prices:
        plt.plot(gbm_vol, control_variate_var[strike_price], label=f"{strike_price}")
    plt.xlabel("GBM Volatility")
    plt.ylabel("Control Variate Estimator Variance")

    plt.legend(title="Strike Price")
    plt.savefig("results/figures/gbm_vol_vs_control_var.png", dpi=600)
    plt.show()
