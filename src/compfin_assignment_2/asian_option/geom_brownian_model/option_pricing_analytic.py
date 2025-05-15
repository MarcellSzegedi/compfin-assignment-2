"""Asian option pricing using Geometric Brownian Motion model."""

import math

import scipy.stats as stats

from compfin_assignment_2.asian_option.model_settings import HestonModelSettings


class OptionPricingAnalytic:
    """CLass to price different options using Geometric Brownian Motion model."""

    def __init__(self, config: HestonModelSettings) -> None:
        """Initialize OptionPricingAnalytic object."""
        self.config = config
        self.adjusted_sigma = math.sqrt(self.config.v_0) / math.sqrt(3)
        self.adjusted_risk_free_rate = (
            (config.risk_free_rate - 0.5 * config.v_0) + self.adjusted_sigma**2
        ) / 2

    @classmethod
    def asian_option_price_calculation(cls, config: HestonModelSettings) -> float:
        """Calculates analytically the asian call option price under GBM."""
        option_pricing = cls(config)
        return option_pricing.asian_option_price_formula()

    def asian_option_price_formula(self):
        """Calculates the present value of the expected payoff of the asian call option."""
        first_term = math.log(self.config.s_0 / self.config.strike)
        denom = math.sqrt(self.config.t_end) * self.adjusted_sigma
        d1 = (
            first_term
            + (self.adjusted_risk_free_rate + 0.5 * self.adjusted_sigma**2) * self.config.t_end
        ) / denom
        d2 = (
            first_term
            + (self.adjusted_risk_free_rate - 0.5 * self.adjusted_sigma**2) * self.config.t_end
        ) / denom
        d1_term = (
            self.config.s_0
            * math.exp(
                (self.adjusted_risk_free_rate - self.config.risk_free_rate) * self.config.t_end
            )
            * stats.norm.cdf(d1)
        )
        d2_term = (
            self.config.strike
            * math.exp(-self.config.risk_free_rate * self.config.t_end)
            * stats.norm.cdf(d2)
        )
        return d1_term - d2_term
