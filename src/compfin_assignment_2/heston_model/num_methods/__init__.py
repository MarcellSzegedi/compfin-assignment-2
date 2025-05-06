"""Collection of numerical methods for solving Heston model."""

from functools import partial

from .numerical_schemes import NumScheme

euler_sim = partial(NumScheme.heston_model_simulation, num_scheme="euler")
milstein_sim = partial(NumScheme.heston_model_simulation, num_scheme="milstein")
