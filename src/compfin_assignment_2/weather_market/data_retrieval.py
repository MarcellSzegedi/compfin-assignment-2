"""
Course: Computational Finance
Names: Marcell Szegedi; Tika van Bennekum; Michael MacFarlane Glasow
Student IDs: 15722635; 13392425; 12317217

File description:
    File gets daily average temperature in Amsterdam from 01-01-2020 till 31-12-2024.
    Data is saved to a csv file.
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry


def get_data():
    """Function gets daily average temperature in Amsterdam from 01-01-2020 till 31-12-2024.
    Data is saved to a csv file."""
    # Setup the Open-Meteo API client with cache and retry
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Define API request parameters
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 52.37,
        "longitude": 4.89,
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "daily": "temperature_2m_mean",
        "timezone": "UTC",
    }

    # Fetch data
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    # Process daily data
    daily = response.Daily()
    daily_temperatures = daily.Variables(0).ValuesAsNumpy()

    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        ),
        "temperature_2m_mean": daily_temperatures,
    }

    # Convert to DataFrame
    daily_dataframe = pd.DataFrame(daily_data)
    daily_dataframe = daily_dataframe.set_index(
        pd.to_datetime(daily_dataframe["date"])
    ).dropna()
    daily_dataframe.to_csv("amsterdam_temperature_data.csv")


get_data()
