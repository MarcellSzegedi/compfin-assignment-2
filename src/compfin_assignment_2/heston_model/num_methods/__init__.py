"""Collection of numerical methods for solving Heston model."""

from .euler_scheme import EulerScheme

euler_sim = EulerScheme.heston_model_simulation

