"""Plot optimal lambda for control variate."""

import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import typer
from tqdm import tqdm

from compfin_assignment_2.asian_option.heston_model.option_pricing import cont_var_coefficient_comp

app = typer.Typer()


@app.command(name="plot-optimal-lambda-cont-var")
def plot_optimal_lambda_cont_var():
    """Plot optimal lambda for control variate."""
    lambda_values = np.linspace(0.5, 1.5, 25)
    strike_prices = np.array([1, 20, 40, 60, 80, 100, 110, 120])
    model_settings = {
        "n_trajectories": 1000,
        "s_0": 100,
        "v_0": 0.2**2,
        "t_end": 1,
        "drift": 0.02,
        "kappa": 6,
        "vol_of_vol": 0.15,
        "stoc_inc_corr": -0.7,
        "num_steps": 10000,
        "risk_free_rate": 0.02,
        "alpha": 0.05,
    }

    variances = defaultdict(list)

    for strike_price in tqdm(strike_prices, desc="Strike Prices", leave=False, position=0):
        for lambda_curr in tqdm(lambda_values, desc="Lambda Values", leave=False, position=1):
            variances[strike_price].append(
                cont_var_coefficient_comp(
                    {**model_settings, "strike": strike_price},
                    gbm_vol=math.sqrt(model_settings["v_0"]),
                    c_coeff=lambda_curr,
                )
            )

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), constrained_layout=True)

    for i, strike_price in enumerate(strike_prices):
        ax = axes[i // 4, i % 4]

        ax.plot(lambda_values, variances[strike_price])
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Estimator Variance")
        ax.set_title(f"Strike Price: {strike_price}")

    plt.savefig("optimal_lambda_cont_var.png", dpi=600)
    plt.show()


plot_optimal_lambda_cont_var()
