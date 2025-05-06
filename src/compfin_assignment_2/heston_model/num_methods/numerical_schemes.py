"""Numerical schemes for solving Heston model."""

import math
from typing import Optional

import numpy as np
import numpy.typing as npt
from tqdm import trange

from compfin_assignment_2.heston_model.num_methods.data_validation import validate_heston_input


class NumScheme:
    """Numerical methods for solving Heston model."""

    def __init__(
        self,
        s_0: float,
        v_0: float,
        t_end: float,
        drift: float,
        kappa: float,
        theta: float,
        vol_of_vol: float,
        stoc_inc_corr: float,
        num_scheme: str,
        step_size: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> None:
        """Initialize EulerScheme object.

        :param s_0: Starting Price of the underlying asset.
        :param v_0: Starting variance of the underlying asset.
        :param t_end: Upper bound of the time interval for the stochastic process simulated.
        :param drift: Drift of the underlying asset.
        :param kappa: The rate at which v reverts to its long term mean (theta).
        :param theta: Long term mean of the volatility.
        :param vol_of_vol: Volatility of volatility.
        :param stoc_inc_corr: Correlation between the stochastic increments of the underlying
                                asset and the volatility process.
        :param num_scheme: Numerical scheme for solving Heston model.
        :param step_size: Step size for Euler scheme.
        :param num_steps: Number of steps for Euler scheme.
        """
        validate_heston_input(
            s_0=s_0,
            v_0=v_0,
            t_end=t_end,
            kappa=kappa,
            theta=theta,
            vol_of_vol=vol_of_vol,
            stoc_inc_corr=stoc_inc_corr,
            num_scheme=num_scheme,
            step_size=step_size,
            num_steps=num_steps,
        )

        self.s_0 = s_0
        self.v_0 = v_0
        self.t_end = t_end
        self.drift = drift
        self.kappa = kappa
        self.theta = theta
        self.vol_of_vol = vol_of_vol
        self.stoc_inc_corr = stoc_inc_corr
        self.num_scheme = num_scheme
        self.step_size = step_size if step_size is not None else t_end / num_steps
        self.num_steps = num_steps if num_steps is not None else int(t_end / self.step_size)

    @classmethod
    def heston_model_simulation(
        cls,
        n_trajectories: int,
        s_0: float,
        v_0: float,
        t_end: float,
        drift: float,
        kappa: float,
        theta: float,
        vol_of_vol: float,
        stoc_inc_corr: float,
        num_scheme: str,
        step_size: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> npt.NDArray[np.float64]:
        """Simulate trajectories of the Heston model using the chosen numerical scheme."""
        model_sim = cls(
            s_0=s_0,
            v_0=v_0,
            t_end=t_end,
            drift=drift,
            kappa=kappa,
            theta=theta,
            vol_of_vol=vol_of_vol,
            stoc_inc_corr=stoc_inc_corr,
            num_scheme=num_scheme,
            step_size=step_size,
            num_steps=num_steps,
        )

        return np.array(
            [
                model_sim.simulate_trajectory()
                for _ in trange(n_trajectories, desc=f"{num_scheme.upper()} scheme simulation")
            ]
        )

    def simulate_trajectory(self, min_var: float = 0) -> list:
        """Simulate a single trajectory of the Heston model using the chosen numerical scheme."""
        s_t = [self.s_0]
        v_t = [self.v_0]
        stochastic_increments = self._simulate_stochastic_increments()

        for i in range(self.num_steps):
            num_scheme_update = self._get_scheme_update_function()

            next_s, next_v = num_scheme_update(
                s_t[-1], v_t[-1], stochastic_increments[i, :], min_var
            )
            s_t.append(next_s)
            v_t.append(next_v)

        return s_t

    def _simulate_stochastic_increments(self) -> npt.NDArray[np.float64]:
        """Simulate correlated stochastic increments for the asset and its variance processes."""
        cov_mat = np.array([[1, self.stoc_inc_corr], [self.stoc_inc_corr, 1]])
        stochastic_increments = np.random.multivariate_normal(
            mean=np.zeros(2), cov=cov_mat, size=self.num_steps
        )
        return stochastic_increments * np.sqrt(self.step_size)

    def _get_scheme_update_function(self) -> callable:
        """Get the function for updating the asset and variance process."""
        match self.num_scheme:
            case "euler":
                return self.euler_scheme_update
            case "milstein":
                return self.milstein_scheme_update
            case _:
                raise ValueError("Invalid numerical scheme.")

    def euler_scheme_update(
        self, curr_s: float, curr_v: float, stoc_inc: npt.NDArray[np.float64], min_var: float
    ) -> tuple[float, float]:
        """Calculates the next asset and the variance process element using Euler scheme."""
        next_s = float(
            curr_s
            + self.drift * curr_s * self.step_size
            + math.sqrt(curr_v) * curr_s * stoc_inc[0]
        )
        next_v = max(
            min_var,
            float(
                curr_v
                + self.kappa * (self.theta - curr_v) * self.step_size
                + self.vol_of_vol * math.sqrt(curr_v) * stoc_inc[1]
            ),
        )
        return next_s, next_v

    def milstein_scheme_update(
        self, curr_s: float, curr_v: float, stoc_inc: npt.NDArray[np.float64], min_var: float
    ) -> tuple[float, float]:
        """Calculates the next asset and the variance process element using Milstein scheme."""
        next_s = float(
            curr_s
            + self.drift * curr_s * self.step_size
            + math.sqrt(curr_v) * curr_s * stoc_inc[0]
            + 0.5 * curr_v * curr_s * (stoc_inc[0] * stoc_inc[0] - self.step_size)
        )
        next_v = max(
            min_var,
            float(
                curr_v
                + self.kappa * (self.theta - curr_v) * self.step_size
                + self.vol_of_vol * math.sqrt(curr_v) * stoc_inc[1]
                + 0.25
                * self.vol_of_vol
                * self.vol_of_vol
                * (stoc_inc[1] * stoc_inc[1] - self.step_size)
            ),
        )
        return next_s, next_v
