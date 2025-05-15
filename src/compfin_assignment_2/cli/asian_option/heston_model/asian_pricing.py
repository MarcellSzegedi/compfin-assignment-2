"""Asian call option pricing using Heston model."""

from collections import defaultdict
from typing import Annotated, DefaultDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import typer
from tqdm import tqdm

from compfin_assignment_2.asian_option.heston_model import (
    asian_option_payoff_euler_sim,
    asian_option_payoff_milstein_sim,
)
from compfin_assignment_2.asian_option.model_settings import HestonModelSettings

app = typer.Typer()


@app.command(name="asian-option-pricing")
def asian_option_pricing(
    n_trajectories: Annotated[
        int, typer.Option("--n-traj", min=1, help="Number of trajectories")
    ] = 10,
) -> None:
    """Asian option pricing using Heston model."""
    model_settings = {
        "n_trajectories": n_trajectories,
        "s_0": 100,
        "v_0": 0.2**2,
        "t_end": 1,
        "drift": 0.02,
        "kappa": 6,
        "theta": 0.1,
        "vol_of_vol": 0.15,
        "stoc_inc_corr": -0.7,
        "num_steps": 10000,
        "risk_free_rate": 0.02,
        "alpha": 0.05,
    }
    strike_prices = np.array([1, 20, 40, 60, 80, 100, 120])

    price_est = defaultdict(lambda: defaultdict(float))
    price_ubound = defaultdict(lambda: defaultdict(float))
    price_lbound = defaultdict(lambda: defaultdict(float))
    price_stde = defaultdict(lambda: defaultdict(float))

    for num_scheme in ["euler", "milstein"]:
        price_est, price_ubound, price_lbound, price_stde = calculate_asian_option_prices(
            num_scheme,
            strike_prices,
            model_settings,
            price_est,
            price_ubound,
            price_lbound,
            price_stde,
        )

    print_table(price_est, "Asian Option Price Point Estimates")
    print_table(price_ubound, f"Asian Option Price Upper Bounds (α={model_settings['alpha']})")
    print_table(price_lbound, f"Asian Option Price Lower Bounds (α={model_settings['alpha']})")
    print_table(price_stde, "Asian Option Price Estimation Standard Error")


def calculate_asian_option_prices(
    numerical_scheme: str,
    strike_prices: npt.NDArray[np.float64],
    config: dict[str, float],
    price_est: DefaultDict[str, DefaultDict[str, float]],
    price_ubound: DefaultDict[str, DefaultDict[str, float]],
    price_lbound: DefaultDict[str, DefaultDict[str, float]],
    price_stde: DefaultDict[str, DefaultDict[str, float]],
) -> tuple[
    DefaultDict[str, DefaultDict[str, float]],
    DefaultDict[str, DefaultDict[str, float]],
    DefaultDict[str, DefaultDict[str, float]],
    DefaultDict[str, DefaultDict[str, float]],
]:
    """Calculates the estimation, c. int. bounds and stde. of the simulated asian option prices."""
    simulator_func = determine_simulation_func(numerical_scheme)
    for strike_price in tqdm(
        strike_prices, desc=f"Simulation Progress: {numerical_scheme.upper()}"
    ):
        price, ub, lb, stde = simulator_func(HestonModelSettings(strike=strike_price, **config))
        price_est[numerical_scheme][strike_price] = price
        price_ubound[numerical_scheme][strike_price] = ub
        price_lbound[numerical_scheme][strike_price] = lb
        price_stde[numerical_scheme][strike_price] = stde

    return price_est, price_ubound, price_lbound, price_stde


def determine_simulation_func(numerical_scheme: str) -> callable:
    """Determines the simulation function based on the chosen numerical scheme."""
    match numerical_scheme:
        case "euler":
            return asian_option_payoff_euler_sim
        case "milstein":
            return asian_option_payoff_milstein_sim
        case _:
            raise ValueError("Invalid numerical scheme.")


def print_table(
    data: dict[str, dict[str, float]],
    table_title: str | None = None,
) -> None:
    """Formats and prints the nested dictionary into a styled pandas DataFrame."""
    df = pd.DataFrame(data).T
    df.index.name = "Numerical Scheme"
    df.columns.name = "Strike Price"

    if table_title:
        print(f"\n{table_title}\n{'=' * len(table_title)}")
    print(df.to_string(float_format="%.4f"))
