"""Entry point of the bencmark CLI app."""

import typer

from .scheme_comp_strong_vanilla import app as strong_conv_app
from .scheme_comp_weak_vanilla import app as weak_conv_app

app = typer.Typer()

app.add_typer(strong_conv_app)
app.add_typer(weak_conv_app)
