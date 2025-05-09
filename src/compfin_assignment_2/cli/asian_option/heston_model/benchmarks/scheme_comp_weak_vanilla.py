"""Command to plot the weak convergence order difference between euler and milstein schemes."""

from typing import Annotated

import numpy as np
import typer
from tqdm import tqdm

from compfin_assignment_2.asian_option.heston_model.benchmarks import (
    weak_convergence_calculation_vanilla_option,
)

from .utils import models_settings, plot_convergence_order_difference

app = typer.Typer()


@app.command(name="weak")
def scheme_comp_strong_vanilla(
    n_trajectories: Annotated[int, typer.Option("--n-traj", min=1, help="Number of trajectories")],
) -> None:
    """Plot the strong convergence order difference between euler and milstein schemes."""
    models_settings["n_trajectories"] = n_trajectories

    n_time_steps = np.power(2, np.arange(7, 15))
    strike_prices = np.array([20, 40, 60, 80, 100, 120, 140, 160])

    euler_results = {}
    milstein_results = {}

    for strike_price in tqdm(strike_prices, desc="Strike Prices", leave=False, position=0):
        euler_payoff_diff, milstein_payoff_diff = weak_convergence_calculation_vanilla_option(
            strike_price, n_time_steps, models_settings
        )

        euler_results[strike_price] = euler_payoff_diff
        milstein_results[strike_price] = milstein_payoff_diff

    plot_convergence_order_difference(
        euler_results,
        milstein_results,
        n_time_steps,
        title="Weak Convergence Order Difference",
        file_name="weak_convergence_plot",
    )
