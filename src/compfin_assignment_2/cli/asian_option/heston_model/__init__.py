"""Heston model CLI entry point."""

import typer

from .asian_pricing import app as asian_option_price_sim_app
from .benchmarks import app as benchmark_app
from .sim_trajectory import app as sim_trajectory_app

app = typer.Typer()

app.add_typer(sim_trajectory_app)
app.add_typer(benchmark_app, name="benchmarks")
app.add_typer(asian_option_price_sim_app)
