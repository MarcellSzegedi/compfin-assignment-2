"""
Course: Computational Finance
Names: Marcell Szegedi; Tika van Bennekum; Michael MacFarlane Glasow
Student IDs: 15722635; 13392425; 12317217

File description:
    File contains model estimations for AR model.
"""

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg


def model_fit(t, a, b, a1, b1):
    """The deterministic model function."""
    omega = 2 * np.pi / 365.25
    return a + b * t + a1 * np.cos(omega * t) + b1 * np.sin(omega * t)


def estimate_parameters():
    """estimated parameters trends data."""
    # Load the saved CSV
    daily_dataframe = pd.read_csv(
        "amsterdam_temperature_data.csv", parse_dates=["date"], index_col="date"
    )
    daily_dataframe = daily_dataframe.dropna(subset=["temperature_2m_mean"])

    # Convert dates to ordinal numbers and fit model
    first_ord = daily_dataframe.index[0].toordinal()
    xdata = np.array([date.toordinal() - first_ord for date in daily_dataframe.index])
    ydata = daily_dataframe["temperature_2m_mean"]

    params_all, cov = curve_fit(model_fit, xdata, ydata, method="lm")
    daily_dataframe["model"] = model_fit(xdata, *params_all)
    daily_dataframe["residuals"] = (
        daily_dataframe["temperature_2m_mean"] - daily_dataframe["model"]
    )
    daily_dataframe.index.freq = "D"

    print("Estimated parameters:", params_all)
    a, b, a1, b1 = params_all
    amplitude = np.sqrt(a1**2 + b1**2)
    shift = np.arctan2(a1, b1) / (2 * np.pi / 365.25)  # shift in days
    print(f"intercept: {a}, trend: {b}, amplitude: {amplitude}, phase shift: {shift}")
    return (daily_dataframe, params_all)


def determine_order(residuals, max_order):
    """Find out what is the best order, the order determines how many previous days are taken into account to
    calculate the value of the current day."""
    best_aic = 10**9
    chosen_order = None

    for order in range(1, max_order + 1):
        model = AutoReg(residuals, lags=order).fit()
        if model.aic < best_aic:
            best_aic = model.aic
            chosen_order = order

    return chosen_order


def ar_model(daily_dataframe):
    """Estimates best order and k for the AR model."""
    residuals = daily_dataframe["residuals"].dropna()

    best_order = determine_order(residuals, 100)
    print("best order:", best_order)

    # Fit an AR model
    ar_model = AutoReg(residuals, lags=best_order).fit()
    gamma = ar_model.params.iloc[1]
    kappa = 1 - gamma
    print("Estimated k (best order):", kappa)

    ar_model = AutoReg(residuals, lags=1).fit()
    gamma = ar_model.params.iloc[1]
    kappa = 1 - gamma
    print("Estimated k (consistent order):", kappa)


def fourier_volatility(t, V, U, *coeffs):
    """
    Fourier series expansion.
    t is time, V is intercept, U is linear trend.
    """

    omega = 2 * np.pi / 365
    half = len(coeffs) // 2

    # Evaluate c_i * sin(i * omega * t)
    c_terms = 0
    for i in range(half):
        ci = coeffs[i]
        c_terms += ci * np.sin((i + 1) * omega * t)

    # Evaluate d_j * cos(j * omega * t)
    d_terms = 0
    for j in range(half):
        dj = coeffs[half + j]
        d_terms += dj * np.cos((j + 1) * omega * t)

    # Fourier series expansion
    return V + U * t + c_terms + d_terms


def fit_fourier_volatility(daily_dataframe, I=2, J=2):
    """
    Fit Fourier series to std of residuals by dates.
    """

    day_of_year = daily_dataframe.index.dayofyear
    residuals = daily_dataframe["residuals"]
    grouped = pd.DataFrame({"day": day_of_year, "residuals": residuals})
    std_per_day = grouped.groupby("day")["residuals"].std()
    std_per_day = std_per_day.reindex(
        range(1, 366)
    ).interpolate()  # Interpolate for missing data

    t = np.array(std_per_day.index)
    y = std_per_day.values

    # Initial guess: V, U, and Fourier coeffsd
    num_coeffs = I + J
    p0 = [1.0, 0.0] + [0.0] * num_coeffs

    # Fit the model
    popt, _ = curve_fit(fourier_volatility, t, y, p0=p0)

    # Return the fitted sigma(t) function
    def sigma_function(day):
        return fourier_volatility(day, *popt)

    return sigma_function


def fourier_plot(daily_dataframe, I=2, J=2):
    """
    Plot the estimated seasonal volatility sigma(t) over a year.
    """
    sigma_func = fit_fourier_volatility(daily_dataframe, I, J)

    t_days = np.arange(1, 366)
    sigma_values = sigma_func(t_days)

    plt.figure(figsize=(10, 4))
    plt.plot(
        t_days, sigma_values, label=r"$\hat{\sigma}(t)$ (Fourier fit)", linewidth=2
    )

    plt.title("Estimated seasonal volatility $\sigma(t)$ using Fourier series", fontsize=16)
    plt.xlabel("Day of Year", fontsize=14)
    plt.ylabel("Volatility", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/fourier_analysis.png", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    daily_dataframe, params_all = estimate_parameters()
    ar_model(daily_dataframe)
    fourier_plot(daily_dataframe)
