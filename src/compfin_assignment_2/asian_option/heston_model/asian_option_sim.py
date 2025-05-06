"""Heston model simulation."""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from compfin_assignment_2.asian_option.heston_model.model_settings import HestonModelSettings
from compfin_assignment_2.asian_option.heston_model.numerical_method import euler_sim, milstein_sim


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
    def estimate_longitudinal_avg_prices(
        cls,
        config: HestonModelSettings,
        numerical_scheme: str,
    ) -> tuple[float, float, float, float]:
        """Estimates the asian call option price under the Heston model.

        Args:
            config: Configuration of the model.
            numerical_scheme: Numerical scheme used for price trajectory simulation.

        Returns:
            Price estimation, confidence interval bounds, and standard error of the estimation.
        """
        asian_pricing = cls(config)
        asian_pricing.set_numerical_scheme(numerical_scheme)

        average_prices = asian_pricing.calculate_longitudinal_avg_prices()
        conf_int_u, conf_int_l, std_err = asian_pricing.compute_conf_interval_long_avg_price(
            average_prices
        )
        return float(np.mean(average_prices)), conf_int_u, conf_int_l, std_err

    def set_numerical_scheme(self, numerical_scheme: str) -> None:
        """Sets the numerical scheme used for price trajectory simulation."""
        match numerical_scheme:
            case "euler":
                self.num_scheme = euler_sim
            case "milstein":
                self.num_scheme = milstein_sim
            case _:
                raise ValueError("Invalid numerical scheme.")

    def calculate_longitudinal_avg_prices(self) -> npt.NDArray[np.float64]:
        """Calculates the average price of the simulated trajectories."""
        sim_result = self.num_scheme(self.config)
        return np.mean(sim_result, axis=0)

    def compute_conf_interval_long_avg_price(
        self,
        average_prices: npt.NDArray[np.float64],
    ) -> tuple[float, float, float]:
        """Computes the confidence interval and standard error of the longitudinal avg prices."""
        std_err = np.std(average_prices, axis=0, ddof=1) / np.sqrt(average_prices.shape[0])
        conf_int_u = average_prices + stats.norm.ppf(1 - self.config.alpha / 2) * std_err
        conf_int_l = average_prices - stats.norm.ppf(1 - self.config.alpha / 2) * std_err

        return conf_int_u, conf_int_l, std_err


settings_example = {
    "n_trajectories": 50,
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
