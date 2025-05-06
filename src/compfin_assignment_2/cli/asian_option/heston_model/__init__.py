"""Heston model CLI entry point."""

import typer

from .sim_trajectory import app as sim_trajectory_app

app = typer.Typer()

app.add_typer(sim_trajectory_app)
