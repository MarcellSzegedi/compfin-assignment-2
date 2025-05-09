"""Methods to price different options based on existing underlying asset trajectory."""

import numpy as np
import numpy.typing as npt

from compfin_assignment_2.asian_option.heston_model.model_settings import HestonModelSettings


class OptionPricingSim:
    """Class to price different options based on existing underlying asset trajectory."""

    def __init__(
        self,
        asset_price: npt.NDArray[np.float64],
        config: HestonModelSettings,
    ) -> None:
        """Initialize OptionPricing object."""
        self.asset_price = asset_price
        self.config = config

    @classmethod
    def vanilla_call_option_payoff_computation(
        cls,
        asset_price: npt.NDArray[np.float64],
        config: HestonModelSettings,
    ) -> npt.NDArray[np.float64]:
        """Calculates the european call option price."""
        option_pricing_instance = cls(asset_price, config)
        payoffs = option_pricing_instance.vanilla_call_option_payoff_calc()
        return payoffs

    @classmethod
    def asian_arithmetic_call_option_payoff_computation(
        cls,
        asset_price: npt.NDArray[np.float64],
        config: HestonModelSettings,
    ) -> npt.NDArray[np.float64]:
        """Calculates the asian call option price."""
        option_pricing_instance = cls(asset_price, config)
        payoffs = option_pricing_instance.asian_arithmetic_call_option_payoff_calc()
        return payoffs

    def vanilla_call_option_payoff_calc(self) -> npt.NDArray[np.float64]:
        """Calculates the payoff of the vanilla call option based on the simulated asset prices.

        The payoff is calculated with Monte-Carlo simulation.
        """
        return np.maximum(self.asset_price[:, -1] - self.config.strike, 0)

    def asian_arithmetic_call_option_payoff_calc(self) -> npt.NDArray[np.float64]:
        """Calculates the payoff of the asian arithmetic call option.

        The payoff is calculated with Monte-Carlo simulation based on the simulated asset prices.
        """
        avg_trajectory_prices = np.mean(self.asset_price, axis=1)
        return np.maximum(avg_trajectory_prices - self.config.strike, 0)
