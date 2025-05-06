"""Plot trajectories."""

from typing import Annotated

import matplotlib.pyplot as plt
import typer

from compfin_assignment_2.asian_option.heston_model.model_settings import HestonModelSettings
from compfin_assignment_2.asian_option.heston_model.numerical_method import euler_sim, milstein_sim

app = typer.Typer()


@app.command(name="sim-trajectory")
def sim_trajectory(
    n_trajectories: Annotated[
        int, typer.Option("--n-traj", min=1, help="Number of trajectories")
    ] = 10,
) -> None:
    """Plot trajectories of the Heston model using Euler and Milstein schemes."""
    settings_example = {
        "n_trajectories": n_trajectories,
        "s_0": 100,
        "v_0": 0.2**2,
        "t_end": 1,
        "drift": 0,
        "kappa": 6,
        "theta": 0.1,
        "vol_of_vol": 0.15,
        "stoc_inc_corr": -0.7,
        "num_steps": 1000,
        "strike_price": 100,
        "risk_free_rate": 0.02,
        "alpha": 0.05,
    }

    model_configuration = HestonModelSettings(**settings_example)

    results_milstain = milstein_sim(model_configuration)
    results_euler = euler_sim(model_configuration)

    plt.figure(figsize=(10, 6))
    for trajectory in results_milstain:
        plt.plot(trajectory, color="red")
    for trajectory in results_euler:
        plt.plot(trajectory, color="blue")
    plt.show()
