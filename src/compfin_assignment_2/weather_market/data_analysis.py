"""
Course: Computational Finance
Names: Marcell Szegedi; Tika van Bennekum; Michael MacFarlane Glasow
Student IDs: 15722635; 13392425; 12317217

File description:
    Data analysis is performed on the daily average temperature in Amsterdam.
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def exploratory_analysis():
    """
    Plots the time series with the mean and variance vizualized.
    """
    # Load the saved CSV
    daily_dataframe = pd.read_csv(
        "amsterdam_temperature_data.csv", parse_dates=["date"], index_col="date"
    )

    plt.figure(figsize=(14, 6))
    # The pure time series
    plt.plot(
        daily_dataframe.index,
        daily_dataframe["temperature_2m_mean"],
        color="gray",
        label="Daily Temp",
        linewidth=0.7,
    )

    # Rolling statistics
    rolling_mean = daily_dataframe["temperature_2m_mean"].rolling(window=30).mean()
    rolling_std = daily_dataframe["temperature_2m_mean"].rolling(window=30).std()

    plt.plot(rolling_mean, color="red", label="30-day Rolling Mean", linewidth=2)
    plt.fill_between(
        rolling_mean.index,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="skyblue",
        alpha=0.3,
        label="±1 variation",
    )

    plt.title(
        "Daily Average Temperature Amsterdam with rolling mean ± rolling variation",
        fontsize=14,
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/time_series.png", dpi=400, bbox_inches="tight")


def decompose_analysis():
    """
    Shows the decomposition of the average daily temperature in Amsterdam.
    The components are the trend, seasonality and residuals.
    """
    # Load the saved CSV
    daily_dataframe = pd.read_csv(
        "amsterdam_temperature_data.csv", parse_dates=["date"], index_col="date"
    )

    # Decompose the time series (assume 365 days per year)
    decompose_result = seasonal_decompose(
        daily_dataframe["temperature_2m_mean"], model="additive", period=365
    )

    # Plots the decomposition with build-in plot from statsmodels
    fig = decompose_result.plot()
    fig.set_size_inches(10, 8)
    fig.suptitle("Decomposition of daily average Temperature in Amsterdam", fontsize=14)
    fig.tight_layout()
    fig.savefig("results/time_series_decomposed.png", dpi=400, bbox_inches="tight")


def residuals():
    """
    Shows the residual trends through a histogram and a Q-Q plot.
    """
    # Load the saved CSV
    daily_dataframe = pd.read_csv(
        "amsterdam_temperature_data.csv", parse_dates=["date"], index_col="date"
    )

    # Decompose the time series (assume 365 days per year)
    decompose_result = seasonal_decompose(
        daily_dataframe["temperature_2m_mean"], model="additive", period=365
    )

    residuals = decompose_result.resid.dropna()

    # The histogram plot of the residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=60)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("results/residuals_histogram.png", dpi=400, bbox_inches="tight")

    # The Q–Q plot of the residuals
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q plot of residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/residuals_qq_plot.png", dpi=400)


def correlation(lags):
    """
    Shows the residual trends through a histogram and a Q-Q plot.
    """
    # Load the saved CSV
    daily_dataframe = pd.read_csv(
        "amsterdam_temperature_data.csv", parse_dates=["date"], index_col="date"
    )

    # Decompose the time series (assume 365 days per year)
    decompose_result = seasonal_decompose(
        daily_dataframe["temperature_2m_mean"], model="additive", period=365
    )

    residuals = decompose_result.resid.dropna()

    # The ACF plot of the residuals
    plt.figure(figsize=(10, 4))
    plot_acf(residuals, lags=lags)
    plt.title("Autocorrelation function (ACF) of residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/residuals_acf.png", dpi=400)

    # The PACF plot of the residuals
    plt.figure(figsize=(10, 4))
    plot_pacf(
        residuals, lags=lags, method="ywm"
    )  # 'ywm' = Yule-Walker Modified (stable method recommend for pacf)
    plt.title("Partial autocorrelation function (PACF) of residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/residuals_pacf.png", dpi=400)


def stationary():
    """
    Tests whetere residuals are stationary.
    """
    # Load the saved CSV
    daily_dataframe = pd.read_csv(
        "amsterdam_temperature_data.csv", parse_dates=["date"], index_col="date"
    )

    # Decompose the time series (assume 365 days per year)
    decompose_result = seasonal_decompose(
        daily_dataframe["temperature_2m_mean"], model="additive", period=365
    )
    residuals = decompose_result.resid.dropna()

    # Perform Augmented Dickey-Fuller test
    result = adfuller(residuals)

    # Print the results
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])


if __name__ == "__main__":
    exploratory_analysis()
    decompose_analysis()
    residuals()
    correlation(
        lags=50
    )  # The lags determines how many days are used in the correlation calculations
    stationary()
