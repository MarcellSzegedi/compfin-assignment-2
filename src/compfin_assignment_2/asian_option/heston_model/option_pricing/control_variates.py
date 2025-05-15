"""Arithmetic Asian Call Option Simulation using control variates method."""

import math
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy.stats as stats

from compfin_assignment_2.asian_option.geom_brownian_model import (
    gbm_analytic_asian_option,
    gbm_sim_asian_option,
)
from compfin_assignment_2.asian_option.heston_model.numerical_method import euler_sim
from compfin_assignment_2.asian_option.model_settings import HestonModelSettings
from compfin_assignment_2.utils.commons import generate_stochastic_increments


class OptionPricingControlVariates:
    """Class to price arithmetic asian call options using control variates method."""

    def __init__(self, config: HestonModelSettings) -> None:
        """Initialize OptionPricingControlVariates object."""
        self.config = config

    @classmethod
    def asian_option_pricing(
        cls, config: dict[str, float | int]
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """Computes the point estimate of the option price and the std err of the option price."""
        instance = cls(HestonModelSettings(theta=config["v_0"], **config))
        stoch_increments = generate_stochastic_increments(
            instance.config.step_size, instance.config.num_steps, instance.config.n_trajectories
        )

        gbm_config, heston_config = instance.setup_gbm_and_heston_model_settings(config)

        gbm_asian_option_prices = gbm_sim_asian_option(gbm_config, stoch_increments)
        heston_asian_option_prices = instance.simulate_asian_option_prices_under_heston(
            heston_config, stoch_increments
        )

        control_variate_option_prices = instance.compute_control_variate_prices(
            gbm_asian_option_prices, heston_asian_option_prices, 1
        )

        con_var_est, con_var_ub, con_var_lb, con_var_std_err = (
            instance.calculate_confidence_interval(control_variate_option_prices)
        )

        heston_est, heston_ub, heston_lb, heston_std_err = instance.calculate_confidence_interval(
            heston_asian_option_prices
        )
        return (
            con_var_est,
            con_var_ub,
            con_var_lb,
            con_var_std_err,
            heston_est,
            heston_ub,
            heston_lb,
            heston_std_err,
        )

    @classmethod
    def compute_differences_between_reg_and_control_variate_methods(
        cls, config: dict[str, float | int]
    ) -> tuple[float, float, float, float, float, float]:
        """Compares the difference between regular and control variate methods.

        Args:
            config: Configuration of the simulation.

        Returns:
            Point estimate of the option price and the standard error of the option price.
        """
        instance = cls(HestonModelSettings(theta=config["v_0"], **config))
        stoch_increments = generate_stochastic_increments(
            instance.config.step_size, instance.config.num_steps, instance.config.n_trajectories
        )

        gbm_config, heston_config = instance.setup_gbm_and_heston_model_settings(config)

        gbm_asian_option_prices = gbm_sim_asian_option(gbm_config, stoch_increments)
        heston_asian_option_prices = instance.simulate_asian_option_prices_under_heston(
            heston_config, stoch_increments
        )

        control_est, control_var, control_std_err = (
            instance.calculate_control_variate_option_price_statistics(
                gbm_asian_option_prices, heston_asian_option_prices
            )
        )

        heston_est, heston_var, heston_std_err = instance.calculate_heston_statistics(
            heston_asian_option_prices
        )

        return control_est, control_var, control_std_err, heston_est, heston_var, heston_std_err

    @classmethod
    def compute_control_variate_estimator_variance(
        cls, config: dict[str, float | int], gbm_vol: float
    ) -> float:
        """Computes the variance term of the control variate method. (COV / VAR)."""
        instance = cls(HestonModelSettings(theta=config["v_0"], **config))
        stoch_increments = generate_stochastic_increments(
            instance.config.step_size, instance.config.num_steps, instance.config.n_trajectories
        )

        gbm_config, heston_config = instance.setup_gbm_and_heston_model_settings(config, gbm_vol)

        gbm_asian_option_prices = gbm_sim_asian_option(gbm_config, stoch_increments)
        heston_asian_option_prices = instance.simulate_asian_option_prices_under_heston(
            heston_config, stoch_increments
        )

        variance_term = instance._calculate_variance_term(
            gbm_asian_option_prices, heston_asian_option_prices
        )

        return variance_term

    @classmethod
    def compute_cont_var_estimator_variance(
        cls, config: dict[str, float | int], gbm_vol: float, c_coeff: float
    ) -> float:
        """Computes the variance term of the control variate method. (COV / VAR)."""
        instance = cls(HestonModelSettings(theta=config["v_0"], **config))
        stoch_increments = generate_stochastic_increments(
            instance.config.step_size, instance.config.num_steps, instance.config.n_trajectories
        )

        gbm_config, heston_config = instance.setup_gbm_and_heston_model_settings(config, gbm_vol)

        gbm_asian_option_prices = gbm_sim_asian_option(gbm_config, stoch_increments)
        heston_asian_option_prices = instance.simulate_asian_option_prices_under_heston(
            heston_config, stoch_increments
        )

        control_variate_option_prices = instance.compute_control_variate_prices(
            gbm_asian_option_prices, heston_asian_option_prices, c_coeff
        )

        return float(
            np.std(control_variate_option_prices, ddof=1)
            / math.sqrt(instance.config.n_trajectories)
        )

    def simulate_option_prices_control_variate_method(
        self,
        gbm_model_settings: HestonModelSettings,
        heston_model_settings: HestonModelSettings,
        stoch_increments: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Simulates the option prices using the control variate method."""
        raise NotImplementedError

    def simulate_asian_option_prices_under_heston(
        self,
        heston_model_settings: HestonModelSettings,
        stochastic_increments: Optional[npt.NDArray[np.float64]] = None,
    ) -> npt.NDArray[np.float64]:
        """Simulates asian option prices under Heston model."""
        trajectories = euler_sim(
            heston_model_settings, stochastic_increments=stochastic_increments
        )
        payoffs = np.maximum(np.mean(trajectories, axis=1) - self.config.strike, 0)
        return payoffs * np.exp(-self.config.risk_free_rate * self.config.t_end)

    @staticmethod
    def setup_gbm_and_heston_model_settings(
        config: dict[str, float | int],
        gbm_vol: Optional[float] = None,
    ) -> tuple[HestonModelSettings, HestonModelSettings]:
        """Sets up the GBM and Heston model settings.

        Theta parameter is expected to be missing from the configuration dictionary.
        """
        gbm_model_settings = (
            HestonModelSettings(theta=gbm_vol**2, **config)
            if gbm_vol is not None
            else HestonModelSettings(theta=config["v_0"], **config)
        )
        heston_model_settings = HestonModelSettings(theta=config["v_0"], **config)
        return gbm_model_settings, heston_model_settings

    @staticmethod
    def _calculate_variance_term(
        gbm_prices: npt.NDArray[np.float64],
        heston_prices: npt.NDArray[np.float64],
    ) -> float:
        """Calculates the variance term of the control variate method. (COV / VAR)."""
        cov = np.cov(gbm_prices, heston_prices, ddof=1)[0, 1]
        var_gbm = np.var(gbm_prices, ddof=1)
        var_heston = np.var(heston_prices, ddof=1)
        return float(var_heston - cov**2 / var_gbm)

    def calculate_control_variate_option_price_statistics(
        self,
        gbm_prices: npt.NDArray[np.float64],
        heston_prices: npt.NDArray[np.float64],
        c_coeff: Optional[float] = None,
    ) -> tuple[float, float, float]:
        """Calculates the control variate estimator."""
        if c_coeff is None:
            c_coeff = self._calculate_optimal_c_coeff(gbm_prices, heston_prices)

        gbm_analytic_mean = gbm_analytic_asian_option(self.config)
        control_variate_est_traj = heston_prices - c_coeff * (gbm_prices - gbm_analytic_mean)

        point_est = np.mean(control_variate_est_traj)
        estimator_var = np.var(control_variate_est_traj, ddof=1)
        std_error = math.sqrt(estimator_var / self.config.n_trajectories)
        return point_est, estimator_var, std_error

    @staticmethod
    def _calculate_optimal_c_coeff(
        gbm_prices: npt.NDArray[np.float64],
        heston_prices: npt.NDArray[np.float64],
    ) -> float:
        """Calculates the optimal c coefficient."""
        return float(np.cov(gbm_prices, heston_prices, ddof=1)[0, 1] / np.var(gbm_prices, ddof=1))

    def calculate_heston_statistics(
        self, heston_prices: npt.NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """Calculates the heston estimator point estimation, estimator variance, and std error."""
        est = np.mean(heston_prices)
        var = np.var(heston_prices, ddof=1)
        std_err = math.sqrt(var / self.config.n_trajectories)
        return est, var, std_err

    def compute_control_variate_prices(
        self,
        gbm_prices: npt.NDArray[np.float64],
        heston_prices: npt.NDArray[np.float64],
        c_coeff: Optional[float] = None,
    ) -> npt.NDArray[np.float64]:
        """Computes the control variate prices."""
        if c_coeff is None:
            c_coeff = self._calculate_optimal_c_coeff(gbm_prices, heston_prices)

        gbm_analytic_mean = gbm_analytic_asian_option(self.config)
        return heston_prices - c_coeff * (gbm_prices - gbm_analytic_mean)

    def calculate_confidence_interval(
        self, prices: npt.NDArray[np.float64]
    ) -> tuple[float, float, float, float]:
        """Calculates the confidence interval."""
        mean = np.mean(prices)
        std_err = np.std(prices, ddof=1) / math.sqrt(len(prices))
        lower_bound = mean - stats.norm.ppf(1 - self.config.alpha / 2) * std_err
        upper_bound = mean + stats.norm.ppf(1 - self.config.alpha / 2) * std_err
        return mean, lower_bound, upper_bound, std_err


settings_example = {
    "n_trajectories": 1000,
    "s_0": 100,
    "v_0": 0.2**2,
    "t_end": 1,
    "drift": 0.02,
    "kappa": 6,
    "vol_of_vol": 0.15,
    "stoc_inc_corr": -0.7,
    "num_steps": 1000,
    "risk_free_rate": 0.02,
    "alpha": 0.05,
    "strike": 100,
}

# cont_est, cont_var, cont_std_err, heston_est, heston_var, heston_std_err = \
#     OptionPricingControlVariates.compute_differences_between_reg_and_control_variate_methods(
#         settings_example
#     )
# print(f"Control variate estimator: {cont_est:.4f} +/- {cont_std_err:.4f}")
# print(f"Heston estimator: {heston_est:.4f} +/- {heston_std_err:.4f}")
