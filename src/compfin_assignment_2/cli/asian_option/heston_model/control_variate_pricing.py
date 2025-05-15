"""Asian call option pricing using Heston model."""

from collections import defaultdict
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from tqdm import tqdm

from compfin_assignment_2.asian_option.heston_model.option_pricing import (
    control_var_option_pricing,
)

app = typer.Typer()


@app.command(name="asian-option-pricing-control-var")
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

    for strike_price in tqdm(strike_prices, desc="Strike Prices"):
        current_settings = {**model_settings, "strike": strike_price}
        (
            con_var_est,
            con_var_ub,
            con_var_lb,
            con_var_std_err,
            heston_est,
            heston_ub,
            heston_lb,
            heston_std_err,
        ) = control_var_option_pricing(current_settings)
        price_est["control_variate"][strike_price] = con_var_est
        price_ubound["control_variate"][strike_price] = con_var_ub
        price_lbound["control_variate"][strike_price] = con_var_lb
        price_stde["control_variate"][strike_price] = con_var_std_err
        price_est["heston"][strike_price] = heston_est
        price_ubound["heston"][strike_price] = heston_ub
        price_lbound["heston"][strike_price] = heston_lb
        price_stde["heston"][strike_price] = heston_std_err

    print_table(price_est, "Asian Option Price Point Estimates")
    print_table(price_ubound, f"Asian Option Price Upper Bounds (α={model_settings['alpha']})")
    print_table(price_lbound, f"Asian Option Price Lower Bounds (α={model_settings['alpha']})")
    print_table(price_stde, "Asian Option Price Estimation Standard Error")


def print_table(
    data: dict[str, dict[str, float]],
    table_title: str | None = None,
) -> None:
    """Formats and prints the nested dictionary into a styled pandas DataFrame."""
    df = pd.DataFrame(data).T
    df.index.name = "Estimator Type"
    df.columns.name = "Strike Price"

    if table_title:
        print(f"\n{table_title}\n{'=' * len(table_title)}")
    print(df.to_string(float_format="%.4f"))
