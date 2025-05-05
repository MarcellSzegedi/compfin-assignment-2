"""Euler scheme for solving Heston model."""

from typing import Optional

import numpy as np
import numpy.typing as npt
import math
from tqdm import trange

from compfin_assignment_2.utils.validation import validate_heston_input


class EulerScheme:
    """Euler numerical method for solving Heston model."""
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
            step_size: Optional[float] = None,
            num_steps: Optional[int] = None
    ) -> None:
        """Initialize EulerScheme object.

        :param s_0: Starting Price of the underlying asset.
        :param v_0: Starting variance of the underlying asset.
        :param t_end: Upper bound of the time interval for the stochastic process simulated.
        :param drift: Drift of the underlying asset.
        :param kappa: The rate at which v reverts to its long term mean (theta).
        :param theta: Long term mean of the volatility.
        :param vol_of_vol: Volatility of volatility.
        :param stoc_inc_corr: Correlation between the stochastic increments of the underlying asset and the volatility
                                process.
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
            step_size=step_size,
            num_steps=num_steps
        )

        self.s_0 = s_0
        self.v_0 = v_0
        self.t_end = t_end
        self.drift = drift
        self.kappa = kappa
        self.theta = theta
        self.vol_of_vol = vol_of_vol
        self.stoc_inc_corr = stoc_inc_corr
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
            step_size: Optional[float] = None,
            num_steps: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        """Simulate trajectories of the Heston model using Euler scheme."""
        euler_scheme = cls(
            s_0=s_0,
            v_0=v_0,
            t_end=t_end,
            drift=drift,
            kappa=kappa,
            theta=theta,
            vol_of_vol=vol_of_vol,
            stoc_inc_corr=stoc_inc_corr,
            step_size=step_size,
            num_steps=num_steps
        )

        return np.array([euler_scheme.simulate_trajectory() for _ in trange(n_trajectories,
                                                                            desc="Euler scheme simulation")])

    def simulate_trajectory(self, min_var: float = 0) -> list:
        """Simulate a single trajectory of the Heston model using Euler scheme."""
        s_t = [self.s_0]
        v_t = [self.v_0]
        stochastic_increments = self._simulate_stochastic_increments()

        for i in range(self.num_steps):
            s_t.append(float(s_t[-1]
                           + self.drift * s_t[-1] * self.step_size
                           + math.sqrt(v_t[-1]) * s_t[-1] * stochastic_increments[i, 0]))
            v_t.append(max(min_var,
                         float(v_t[-1]
                               + self.kappa * (self.theta - v_t[-1]) * self.step_size
                               + self.vol_of_vol * math.sqrt(v_t[-1]) * stochastic_increments[i, 1])))
        return s_t

    def _simulate_stochastic_increments(self) -> npt.NDArray[np.float64]:
        """Simulate correlated stochastic increments of the underlying asset and the volatility processes."""
        cov_mat = np.array([[1, self.stoc_inc_corr], [self.stoc_inc_corr, 1]])
        stochastic_increments = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=cov_mat,
            size=self.num_steps
        )
        return stochastic_increments * np.sqrt(self.step_size)
