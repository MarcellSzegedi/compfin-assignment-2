"""Function used by multiple modules in the 'compfin' package."""

import math

import numpy as np
import numpy.typing as npt


def generate_stochastic_increments(
    time_step: float, num_steps: int, n_trajectories: int
) -> npt.NDArray[np.float64]:
    """Generates stochastic increments for the asset.

    Args:
        time_step: Length of time step. (delta t)
        n_trajectories: Number of trajectories to simulate.
        num_steps: Number or time steps in one trajectory.

    Returns:
        Stochastic increments. (1D numpy array)
    """
    return np.random.normal(0, math.sqrt(time_step), size=(n_trajectories, num_steps))
