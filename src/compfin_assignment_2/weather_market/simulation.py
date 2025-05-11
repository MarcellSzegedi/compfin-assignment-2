"""
Course: Computational Finance
Names: Marcell Szegedi; Tika van Bennekum; Michael MacFarlane Glasow
Student IDs: 15722635; 13392425; 12317217

File description:
    Simulation of weather derivative market.
"""

import numpy as np
import matplotlib.pyplot as plt
from model import estimate_parameters
from model import model_fit
from model import fit_fourier_volatility


def euler_simulation(nr_days, params_all, kappa, sigma_func, ini_temp):
    """
    Simulate temperature paths using Euler discretization.
    """
    a, b, a1, b1 = params_all
    t = np.arange(nr_days)
    seasonal_mean_values = model_fit(t, a, b, a1, b1)

    # Create the temperate paths
    T_noise = np.zeros(nr_days)
    T_total = np.zeros(nr_days)

    # On day one the path starts with the intitial temperature
    T_noise[0] = ini_temp - seasonal_mean_values[0]
    T_total[0] = ini_temp

    # Now we generate the temperate each day based on the previous day
    for day in range(1, nr_days):
        z = np.random.normal(0, 1)
        day_of_year = day % 365 if day % 365 != 0 else 365
        sigma_today = sigma_func(day_of_year)
        T_noise[day] = T_noise[day - 1] - kappa * T_noise[day - 1] + sigma_today * z
        T_total[day] = T_noise[day] + seasonal_mean_values[day]

    return T_total


def multiple_paths(nr_days, params_all, kappa, sigma_func, ini_temp, nr_paths):
    """
    Because the generation of a path is stochastic, we simulate multiple.
    """
    all_paths = np.zeros((nr_paths, nr_days))
    for i in range(nr_paths):
        all_paths[i] = euler_simulation(
            nr_days, params_all, kappa, sigma_func, ini_temp
        )
    return all_paths


def plot_paths():
    """Plots multiple temperature paths in a plot."""
    daily_dataframe, params_all = estimate_parameters()
    ini_temp = daily_dataframe["temperature_2m_mean"].iloc[0]

    # Parameters
    nr_days = 365
    nr_paths = 5
    kappa = 0.21  # Value estimated in model.py
    sigma_func = fit_fourier_volatility(daily_dataframe)

    all_paths = multiple_paths(
        nr_days, params_all, kappa, sigma_func, ini_temp, nr_paths
    )
    mean_path = np.mean(all_paths, axis=0)
    t = np.arange(nr_days)

    plt.figure(figsize=(12, 6))
    for i in range(nr_paths):
        plt.plot(t, all_paths[i], label=f"Path {i+1}")

    plt.plot(t, mean_path, color="black", linewidth=2, label="Mean Path")
    plt.title("Simulated Temperature Paths with Seasonal Trend")
    plt.xlabel("Day")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/temperature_paths.png", dpi=400, bbox_inches="tight")


def payoff_function(N, K, T, floor_true=False):
    """Basic HDD call/put option payoff."""
    H_n = np.sum(np.maximum(18 - T, 0))  # HDD index, also called DD in other formula

    if floor_true:
        return max(N * (K - H_n), 0)  # Floor
    else:
        return max(N * (H_n - K), 0)  # Ceiling


def call_cap(alpha, K, T, cap):
    """Call option with Cap.
    Is the payoff with a bussines restraint."""
    base_payoff = payoff_function(alpha, K, T)
    elaborated_payoff = min(base_payoff, cap)
    return elaborated_payoff


def put_floor(alpha, K, T, floor):
    """Put option with Floor.
    Applies a minimum guaranteed payoff."""
    base_payoff = payoff_function(alpha, K, T, True)
    elaborated_payoff = min(base_payoff, floor)
    return elaborated_payoff


def collar(alpha, K1, beta, K2, T, cap, floor):
    """Collar option: Long call with cap and short put with floor."""
    base_payoff_call = payoff_function(alpha, K1, T)
    elaborated_payoff_call = min(base_payoff_call, cap)

    base_payoff_put = payoff_function(beta, K2, T, True)
    elaborated_payoff_put = min(base_payoff_put, floor)

    collar = elaborated_payoff_call - elaborated_payoff_put
    return collar


def payoff_histogram():
    """Plots payoff histogram."""

    daily_dataframe, params_all = estimate_parameters()
    ini_temp = daily_dataframe["temperature_2m_mean"].iloc[0]

    # Parameters
    nr_paths = 10000
    nr_days = 365
    kappa = 0.21  # Use estimated value
    sigma_func = fit_fourier_volatility(daily_dataframe)
    temperature_paths = multiple_paths(
        nr_days, params_all, kappa, sigma_func, ini_temp, nr_paths
    )

    # More parameter choices
    strike_K = 2500
    cap = 440
    alpha = 1.0

    payoffs = [call_cap(alpha, strike_K, T, cap) for T in temperature_paths]

    plt.figure(figsize=(8, 5))
    plt.hist(payoffs, bins=50, edgecolor="black")
    plt.title(f"Histogram of HDD Call Payoffs (K = {strike_K})")
    plt.xlabel("Payoff")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("results/hdd_call_histogram.png", dpi=300, bbox_inches="tight")


def payoff_comparison():
    """Plots payoff comparison for three different constrained financial contracts."""

    daily_dataframe, params_all = estimate_parameters()
    ini_temp = daily_dataframe["temperature_2m_mean"].iloc[0]

    # Parameters
    nr_paths = 10000
    nr_days = 365
    kappa = 0.21  # Use estimated value
    sigma_func = fit_fourier_volatility(daily_dataframe)
    temperature_paths = multiple_paths(
        nr_days, params_all, kappa, sigma_func, ini_temp, nr_paths
    )

    # More parameters
    strikes = np.arange(2200, 3000, 50)
    alpha = 1.0
    beta = 1.0
    cap_ratio = 440 / 2500
    floor_ratio = 60 / 2500
    r = 0.05
    discount = np.exp(-r * 1)

    call_prices, put_prices, collar_prices = [], [], []

    # We compare a range of strikes
    for K1 in strikes:
        K2 = K1 * 0.9
        cap = cap_ratio * K1
        floor = floor_ratio * K2
        call_payoffs = []
        put_payoffs = []
        collar_payoffs = []

        for T in temperature_paths:
            call_payoffs.append(call_cap(alpha, K1, T, cap))
            put_payoffs.append(put_floor(alpha, K1, T, floor))
            collar_payoffs.append(collar(alpha, K1, beta, K2, T, cap, floor))

        call_prices.append(discount * np.mean(call_payoffs))
        put_prices.append(discount * np.mean(put_payoffs))
        collar_prices.append(discount * np.mean(collar_payoffs))

    plt.figure(figsize=(10, 6))
    plt.plot(strikes, call_prices, label="Call with Cap")
    plt.plot(strikes, put_prices, label="Put with Floor")
    plt.plot(strikes, collar_prices, label="Collar")
    plt.title("Option Prices vs Strike")
    plt.xlabel("Strike (K)")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/payoff_comparison.png", dpi=400, bbox_inches="tight")


def estimate_cap_floor(
    temperature_paths, strike, alpha=1.0, beta=1.0, percentile=95, buffer=1.1
):
    """This was just a function in the background to help determine good cap/floor values."""
    HDDs = np.array([np.sum(np.maximum(18 - T, 0)) for T in temperature_paths])

    call_payoffs = np.maximum(alpha * (HDDs - strike), 0)
    cap = round(np.percentile(call_payoffs, percentile) * buffer, 2)

    put_payoffs = np.maximum(beta * (strike - HDDs), 0)
    floor = round(np.percentile(put_payoffs, percentile) * buffer, 2)

    return cap, floor


if __name__ == "__main__":
    plot_paths()
    payoff_comparison()
    payoff_histogram()
