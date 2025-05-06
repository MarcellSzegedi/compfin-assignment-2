"""Heston model simulation."""

import matplotlib.pyplot as plt

from compfin_assignment_2.heston_model.numerical_method import euler_sim, milstein_sim
from compfin_assignment_2.heston_model.model_settings import HestonModelSettings

settings_example = {
    "n_trajectories": 50,
    "s_0": 100,
    "v_0": 0.2**2,
    "t_end": 1,
    "drift": 0,
    "kappa": 6,
    "theta": 0.1,
    "vol_of_vol": 0.15,
    "stoc_inc_corr": -0.7,
    "num_steps": 1000,
}

model_configuration = HestonModelSettings(**settings_example)

results_milstain = milstein_sim(model_configuration)
results_euler = euler_sim(model_configuration)

plt.figure(figsize=(10, 6))
for trajectory in results_milstain:
    plt.plot(trajectory, color="red")
for trajectory in results_euler:
    plt.plot(trajectory, color="blue")
plt.show()
