"""Methods to price different options analytically."""

import math
from typing import Optional

import numpy as np
import numpy.typing as npt

from compfin_assignment_2.asian_option.model_settings import HestonModelSettings
from compfin_assignment_2.utils.commons import generate_stochastic_increments


class OptionPricingAnalytical:
    """Class to price different options analytically."""

    def __init__(
        self,
        config: HestonModelSettings,
        stoch_increments: Optional[npt.NDArray[np.float64]] = None,
    ) -> None:
        """Initialize OptionPricing object."""
        if stoch_increments is not None:
            assert stoch_increments.shape[0] == config.n_trajectories
            assert stoch_increments.shape[1] == config.num_steps

        self.config = config
        self.stoch_increments = (
            stoch_increments
            if stoch_increments is not None
            else generate_stochastic_increments(
                config.step_size, config.num_steps, config.n_trajectories
            )
        )

    @classmethod
    def vanilla_call_option_payoff_computation(
        cls,
        config: HestonModelSettings,
        stoch_increments: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Calculates the analytical solution for the vanilla call option."""
        option_pricing = cls(config, stoch_increments)
        asset_prices = option_pricing.asset_price_calc()
        payoffs = option_pricing.vanilla_call_option_payoff_calc(asset_prices)
        return payoffs

    def asset_price_calc(self) -> npt.NDArray[np.float64]:
        """Calculates the final asset price at time T."""
        wiener_increments = np.cumsum(self.stoch_increments, axis=1)
        times = np.linspace(0, self.config.t_end, self.config.num_steps + 1)[1:]

        prices = np.array(
            [
                self.config.s_0
                * np.exp(
                    (self.config.drift - 0.5 * self.config.v_0) * times
                    + math.sqrt(self.config.v_0) * wiener_increment
                )
                for wiener_increment in wiener_increments
            ]
        )

        return np.hstack((self.config.s_0 * np.ones((self.config.n_trajectories, 1)), prices))

    def vanilla_call_option_payoff_calc(
        self, asset_prices: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculates the payoff of the vanilla call option."""
        return np.maximum(asset_prices[:, -1] - self.config.strike, 0)
