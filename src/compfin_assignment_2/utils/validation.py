"""Validate and control the input data."""

from typing import Optional

def validate_heston_input(
        s_0: float,
        v_0: float,
        t_end: float,
        kappa: float,
        theta: float,
        vol_of_vol: float,
        stoc_inc_corr: float,
        step_size: Optional[float] = None,
        num_steps: Optional[int] = None
) -> None:
    """Validate the input data for Heston model."""
    if step_size is None and num_steps is None:
        raise ValueError("Either step_size or num_steps should be specified.")
    if step_size is not None and num_steps is not None:
        raise ValueError("Only one of step_size or num_steps should be specified.")
    if step_size is not None and step_size <= 0:
        raise ValueError("step_size should be positive.")
    if num_steps is not None and num_steps <= 0:
        raise ValueError("num_steps should be positive.")
    if s_0 <= 0:
        raise ValueError("Starting asset price 's_0' should be positive.")
    if v_0 <= 0:
        raise ValueError("Starting variance of the underlying asset 'v_0' should be positive.")
    if t_end <= 0:
        raise ValueError("t_end should be positive.")
    if kappa < 0:
        raise ValueError("kappa should be non-negative.")
    if theta < 0:
        raise ValueError("theta should be non-negative.")
    if vol_of_vol < 0:
        raise ValueError("vol_of_vol should be non-negative.")
    if stoc_inc_corr < -1 or stoc_inc_corr > 1:
        raise ValueError("stoc_inc_corr should be in [-1, 1].")
