"""Heston model simulation."""

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from compfin_assignment_2.asian_option.heston_model.numerical_method import euler_sim, milstein_sim
from compfin_assignment_2.asian_option.model_settings import HestonModelSettings


class AsianOptionSim:
    """Asian Option Price simulation under Heston model."""

    def __init__(self, config: HestonModelSettings):
        """Initializes an asian option simulation object.

        Args:
            config: Configuration of the model.
        """
        self.config = config
        self.num_scheme = None

    @classmethod
    def estimate_asian_option_price(
        cls, config: HestonModelSettings, numerical_scheme: str
    ) -> tuple[float, float, float, float]:
        """Estimates the price of an asian call option under the Heston model.

        Args:
            config: Configuration of the model.
            numerical_scheme: Numerical scheme used for underlying asset trajectory simulation.

        Returns:
            Point estimation, confidence interval bounds and standard error of the asian option
            price.
        """
        asian_option_sim = cls(config)
        asian_option_sim.set_numerical_scheme(numerical_scheme)

        asian_payoffs = asian_option_sim.simulate_asian_payoffs()
        discounted_asian_payoffs = asian_payoffs * np.exp(
            -asian_option_sim.config.risk_free_rate * asian_option_sim.config.t_end
        )
        conf_int_u, conf_int_l, std_err = asian_option_sim.compute_conf_interval_long_avg_price(
            discounted_asian_payoffs
        )

        return float(np.mean(discounted_asian_payoffs)), conf_int_u, conf_int_l, std_err

    def simulate_asian_payoffs(self) -> npt.NDArray[np.float64]:
        """Simulates asian call option payoffs under the Heston model."""
        sim_result = self.num_scheme(self.config)
        avg_prices = np.mean(sim_result, axis=0)
        return np.maximum(avg_prices - self.config.strike, 0)

    def set_numerical_scheme(self, numerical_scheme: str) -> None:
        """Sets the numerical scheme used for price trajectory simulation."""
        match numerical_scheme:
            case "euler":
                self.num_scheme = euler_sim
            case "milstein":
                self.num_scheme = milstein_sim
            case _:
                raise ValueError("Invalid numerical scheme.")

    def compute_conf_interval_long_avg_price(
        self,
        measurements: npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        """Computes the confidence interval and standard error of the given measurements."""
        std_err = np.std(measurements, axis=0, ddof=1) / np.sqrt(measurements.shape[0])
        conf_int_u = np.mean(measurements) + stats.norm.ppf(1 - self.config.alpha / 2) * std_err
        conf_int_l = np.mean(measurements) - stats.norm.ppf(1 - self.config.alpha / 2) * std_err

        return conf_int_u, conf_int_l, std_err
