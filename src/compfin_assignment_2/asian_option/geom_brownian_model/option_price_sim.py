"""Simulate geometric asian call option prices under GBM."""

import math
from typing import Optional

import numpy as np
import numpy.typing as npt

from compfin_assignment_2.asian_option.model_settings import HestonModelSettings
from compfin_assignment_2.utils.commons import generate_stochastic_increments


class SimulateGAO:
    """Simulates geometric asian option prices under GBM."""

    def __init__(
        self,
        config: HestonModelSettings,
        stochastic_increments: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Initialize SimulateGAO object."""
        if stochastic_increments is not None:
            assert stochastic_increments.shape[0] == config.n_trajectories
            assert stochastic_increments.shape[1] == config.num_steps

        self.config = config
        self.stochastic_increments = (
            stochastic_increments
            if stochastic_increments is not None
            else generate_stochastic_increments(
                config.step_size, config.num_steps, config.n_trajectories
            )
        )

    @classmethod
    def simulate_asian_option_prices(
        cls,
        config: HestonModelSettings,
        stochastic_increments: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Simulates asian option prices under GBM using geometric euler scheme."""
        asian_option_price = cls(config, stochastic_increments)

        price_trajectories = asian_option_price.simulate_price_trajectories()
        discounted_payoffs = asian_option_price.compute_discounted_payoffs(price_trajectories)

        return discounted_payoffs

    def simulate_price_trajectories(self) -> npt.NDArray[np.float64]:
        """Simulates price trajectories under GBM using geometric euler scheme."""
        drift = (self.config.drift - 0.5 * self.config.theta) * self.config.step_size
        diffusion = math.sqrt(self.config.theta) * self.stochastic_increments
        log_return = drift + diffusion
        log_increments = np.cumsum(log_return, axis=1)
        price_trajectories = self.config.s_0 * np.exp(log_increments)
        return np.concatenate(
            (self.config.s_0 * np.ones((self.config.n_trajectories, 1)), price_trajectories),
            axis=1,
        )

    def compute_discounted_payoffs(
        self, price_trajectories: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Computes discounted payoffs of the asian call option."""
        geom_avg = np.exp(np.mean(np.log(price_trajectories), axis=1))
        payoffs = np.maximum(geom_avg - self.config.strike, 0)
        discounted_payoffs = np.exp(-self.config.risk_free_rate * self.config.t_end) * payoffs
        return discounted_payoffs
