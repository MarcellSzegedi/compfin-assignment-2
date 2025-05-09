"""Plot trajectories."""

from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import typer

from compfin_assignment_2.asian_option.heston_model.model_settings import HestonModelSettings
from compfin_assignment_2.asian_option.heston_model.numerical_method import euler_sim, milstein_sim

app = typer.Typer()


@app.command(name="sim-trajectory")
def sim_trajectory(
    n_trajectories: Annotated[
        int, typer.Option("--n-traj", min=1, help="Number of trajectories")
    ] = 10,
    logging: Annotated[bool, typer.Option("--logging", help="Logging")] = True,
) -> None:
    """Plot trajectories of the Heston model using Euler and Milstein schemes."""
    settings_example = {
        "n_trajectories": n_trajectories,
        "s_0": 100,
        "v_0": 0.2**2,
        "t_end": 1,
        "drift": 0.02,
        "kappa": 6,
        "theta": 0.1,
        "vol_of_vol": 0.15,
        "stoc_inc_corr": -0.7,
        "num_steps": 1000,
        "risk_free_rate": 0.02,
        "alpha": 0.05,
        "strike": 100,
    }

    model_configuration = HestonModelSettings(**settings_example)

    results_milstein = milstein_sim(model_configuration, logging=logging)
    results_euler = euler_sim(model_configuration, logging=logging)

    plt.figure(figsize=(10, 6))
    for trajectory in results_euler:
        plt.plot(trajectory, color="blue", alpha=0.3, linewidth=0.2)
    for trajectory in results_milstein:
        plt.plot(trajectory, color="red", alpha=0.3, linewidth=0.2)

    plt.plot(np.mean(results_milstein, axis=0), color="red", label="Milstein", linewidth=2)
    plt.plot(np.mean(results_euler, axis=0), color="blue", label="Euler", linewidth=2)
    plt.legend()
    plt.show()
