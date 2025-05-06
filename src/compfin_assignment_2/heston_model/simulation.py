"""Heston model simulation."""

import matplotlib.pyplot as plt

from compfin_assignment_2.heston_model.num_methods import euler_sim, milstein_sim

results_milstain = milstein_sim(
    n_trajectories=50,
    s_0=100,
    v_0=0.2**2,
    t_end=1,
    drift=0,
    kappa=6,
    theta=0.1,
    vol_of_vol=0.15,
    stoc_inc_corr=-0.7,
    num_steps=1000,
)

results_euler = euler_sim(
    n_trajectories=50,
    s_0=100,
    v_0=0.2**2,
    t_end=1,
    drift=0,
    kappa=6,
    theta=0.1,
    vol_of_vol=0.15,
    stoc_inc_corr=-0.7,
    num_steps=1000,
)

plt.figure(figsize=(10, 6))
for trajectory in results_milstain:
    plt.plot(trajectory, color="red")
for trajectory in results_euler:
    plt.plot(trajectory, color="blue")
plt.show()
