"""Contains the benchmarking functions for the Euler - Milstein method comparison."""

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from compfin_assignment_2.asian_option.heston_model.model_settings import HestonModelSettings
from compfin_assignment_2.asian_option.heston_model.numerical_method import euler_sim, milstein_sim
from compfin_assignment_2.asian_option.heston_model.option_pricing import (
    vanilla_option_payoff_analytical,
    vanilla_option_payoff_sim,
)
from compfin_assignment_2.utils.commons import generate_stochastic_increments


def strong_convergence_calculation_vanilla_option(
    strike_price: float,
    n_time_steps: npt.NDArray[np.int64],
    models_settings: dict[str, float | int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates the strong convergence error for euler and milstein schemes."""
    euler_simulated_payoff_diff = []
    milstein_simulated_payoff_diff = []

    for time_step in tqdm(n_time_steps, desc=f"K: {strike_price}", leave=False, position=1):
        true_payoffs, euler_payoffs, milstein_payoffs = calculate_payoffs(
            strike_price, time_step, models_settings
        )

        euler_simulated_payoff_diff.append(np.mean(np.abs(euler_payoffs - true_payoffs)))
        milstein_simulated_payoff_diff.append(np.mean(np.abs(milstein_payoffs - true_payoffs)))

    return np.array(euler_simulated_payoff_diff), np.array(milstein_simulated_payoff_diff)


def weak_convergence_calculation_vanilla_option(
    strike_price: float,
    n_time_steps: npt.NDArray[np.int64],
    models_settings: dict[str, float | int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates the weak convergence error for euler and milstein schemes."""
    euler_simulated_payoff_diff = []
    milstein_simulated_payoff_diff = []

    for time_step in tqdm(n_time_steps, desc=f"K: {strike_price}", leave=False, position=1):
        true_payoffs, euler_payoffs, milstein_payoffs = calculate_payoffs(
            strike_price, time_step, models_settings
        )

        euler_simulated_payoff_diff.append(np.abs(np.mean(euler_payoffs) - np.mean(true_payoffs)))
        milstein_simulated_payoff_diff.append(
            np.abs(np.mean(milstein_payoffs) - np.mean(true_payoffs))
        )

    return np.array(euler_simulated_payoff_diff), np.array(milstein_simulated_payoff_diff)


def calculate_payoffs(
    strike_price: float,
    current_n_time_steps: int,
    models_settings: dict[str, float | int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates the true payoffs besides the payoffs for the euler and milstein schemes."""
    current_settings = HestonModelSettings(
        num_steps=current_n_time_steps, strike=strike_price, **models_settings
    )
    asset_stoch_increments = generate_stochastic_increments(
        current_settings.step_size, current_settings.num_steps, current_settings.n_trajectories
    )

    euler_trajectories = euler_sim(current_settings, stochastic_increments=asset_stoch_increments)
    milstein_trajectories = milstein_sim(
        current_settings, stochastic_increments=asset_stoch_increments
    )

    euler_payoffs = vanilla_option_payoff_sim(euler_trajectories, current_settings)
    milstein_payoffs = vanilla_option_payoff_sim(milstein_trajectories, current_settings)
    true_payoffs = vanilla_option_payoff_analytical(current_settings, asset_stoch_increments)

    return true_payoffs, euler_payoffs, milstein_payoffs
