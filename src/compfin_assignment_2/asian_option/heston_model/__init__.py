"""Heston model __init__.py file."""

from functools import partial

from .asian_option_sim import AsianOptionSim

asian_option_payoff_euler_sim = partial(
    AsianOptionSim.estimate_asian_option_price, numerical_scheme="euler"
)
asian_option_payoff_milstein_sim = partial(
    AsianOptionSim.estimate_asian_option_price, numerical_scheme="milstein"
)
