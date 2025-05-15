"""Heston model CLI entry point."""

import typer

from .asian_pricing import app as asian_option_price_sim_app
from .benchmarks import app as benchmark_app
from .control_variate_pricing import app as control_variate_pricing_app
from .control_vs_heston_stat import app as control_vs_heston_stat_app
from .gbm_vol_vs_control_var import app as gbm_vol_app
from .sim_trajectory import app as sim_trajectory_app

app = typer.Typer()

app.add_typer(sim_trajectory_app)
app.add_typer(benchmark_app, name="benchmarks")
app.add_typer(asian_option_price_sim_app)
app.add_typer(gbm_vol_app)
app.add_typer(control_vs_heston_stat_app)
app.add_typer(control_variate_pricing_app)
