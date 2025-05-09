"""CLI entry point."""

import typer

from .asian_option.heston_model import app as heston_app

app = typer.Typer()

app.add_typer(heston_app, name="heston-model")
