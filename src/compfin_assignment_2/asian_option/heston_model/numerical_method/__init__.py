"""Collection of numerical methods for solving Heston model."""

from functools import partial

from compfin_assignment_2.asian_option.heston_model.numerical_method.numerical_schemes import (
    NumScheme,
)

euler_sim = partial(NumScheme.heston_model_simulation, numerical_scheme="euler")
milstein_sim = partial(NumScheme.heston_model_simulation, numerical_scheme="milstein")
gbm_sim = partial(NumScheme.gbm_model_simulation)
