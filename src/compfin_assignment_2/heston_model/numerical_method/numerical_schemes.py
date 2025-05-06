"""Numerical schemes for solving Heston model."""

import math

import numpy as np
import numpy.typing as npt
from tqdm import trange

from compfin_assignment_2.heston_model.model_settings import HestonModelSettings


class NumScheme:
    """Numerical methods for solving Heston model."""

    def __init__(self, config: HestonModelSettings) -> None:
        """Initialize EulerScheme object."""
        self.config = config
        self.num_scheme = None

    @classmethod
    def heston_model_simulation(
        cls, config: HestonModelSettings, numerical_scheme: str
    ) -> npt.NDArray[np.float64]:
        """Simulate trajectories of the Heston model using the chosen numerical scheme."""
        model_sim = cls(config)
        model_sim.set_numerical_scheme(numerical_scheme)

        return np.array(
            [
                model_sim.simulate_trajectory()
                for _ in trange(
                    model_sim.config.n_trajectories,
                    desc=f"{numerical_scheme.upper()} scheme simulation",
                )
            ]
        )

    def simulate_trajectory(self, min_var: float = 0) -> list:
        """Simulate a single trajectory of the Heston model using the chosen numerical scheme."""
        s_t = [self.config.s_0]
        v_t = [self.config.v_0]
        stochastic_increments = self._simulate_stochastic_increments()

        for i in range(self.config.num_steps):
            next_s, next_v = self.num_scheme(
                s_t[-1], v_t[-1], stochastic_increments[i, :], min_var
            )
            s_t.append(next_s)
            v_t.append(next_v)

        return s_t

    def _simulate_stochastic_increments(self) -> npt.NDArray[np.float64]:
        """Simulate correlated stochastic increments for the asset and its variance processes."""
        cov_mat = np.array([[1, self.config.stoc_inc_corr], [self.config.stoc_inc_corr, 1]])
        stochastic_increments = np.random.multivariate_normal(
            mean=np.zeros(2), cov=cov_mat, size=self.config.num_steps
        )
        return stochastic_increments * np.sqrt(self.config.step_size)

    def set_numerical_scheme(self, numerical_scheme: str) -> None:
        """Get the function for updating the asset and variance process."""
        match numerical_scheme:
            case "euler":
                self.num_scheme = self.euler_scheme_update
            case "milstein":
                self.num_scheme = self.milstein_scheme_update
            case _:
                raise ValueError("Invalid numerical scheme.")

    def euler_scheme_update(
        self, curr_s: float, curr_v: float, stoc_inc: npt.NDArray[np.float64], min_var: float
    ) -> tuple[float, float]:
        """Calculates the next asset and the variance process element using Euler scheme."""
        next_s = float(
            curr_s
            + self.config.drift * curr_s * self.config.step_size
            + math.sqrt(curr_v) * curr_s * stoc_inc[0]
        )
        next_v = max(
            min_var,
            float(
                curr_v
                + self.config.kappa * (self.config.theta - curr_v) * self.config.step_size
                + self.config.vol_of_vol * math.sqrt(curr_v) * stoc_inc[1]
            ),
        )
        return next_s, next_v

    def milstein_scheme_update(
        self, curr_s: float, curr_v: float, stoc_inc: npt.NDArray[np.float64], min_var: float
    ) -> tuple[float, float]:
        """Calculates the next asset and the variance process element using Milstein scheme."""
        next_s = float(
            curr_s
            + self.config.drift * curr_s * self.config.step_size
            + math.sqrt(curr_v) * curr_s * stoc_inc[0]
            + 0.5 * curr_v * curr_s * (stoc_inc[0] * stoc_inc[0] - self.config.step_size)
        )
        next_v = max(
            min_var,
            float(
                curr_v
                + self.config.kappa * (self.config.theta - curr_v) * self.config.step_size
                + self.config.vol_of_vol * math.sqrt(curr_v) * stoc_inc[1]
                + 0.25
                * self.config.vol_of_vol
                * self.config.vol_of_vol
                * (stoc_inc[1] * stoc_inc[1] - self.config.step_size)
            ),
        )
        return next_s, next_v
