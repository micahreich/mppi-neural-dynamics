import sys
sys.path.append("..")

from system_controller import SystemController, ControlNoiseInit
from utils import transform_angle_error, transform_02pi_to_negpospi, Simulator, Path, Trajectory

import numpy as np
import time
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.integrate import odeint

plt.style.use(["science", "grid"])

from systems.dynamical_systems import Pendulum


pend = Pendulum(m=1, l=1, b=0.1, dt=1 / 10)

DESIRED_THETA = np.pi
DESIRED_OMEGA = 0.0


def terminal_cost(x):
    theta, omega = x

    theta_error = np.cos(theta - DESIRED_THETA + np.pi) + 1
    omega_error = DESIRED_OMEGA - omega

    error = np.array([theta_error, omega_error])
    Q = np.diag([1000, 1])

    return error.T @ Q @ error


def state_cost(x):
    theta, omega = x

    theta_error = np.cos(theta - DESIRED_THETA + np.pi) + 1

    error = np.array([theta_error])
    Q = np.diag([1000])

    return error.T @ Q @ error


pend_controller = SystemController(
    ds=pend,
    n_rollouts=100,
    horizon_length=10,
    exploration_cov=np.diag([2.0 ** 2]),
    exploration_lambda=1,
    alpha_mu=0.5,
    alpha_sigma=0.95,
    state_cost=state_cost, terminal_cost=terminal_cost,
    control_range={"min": pend.tau_lo, "max": pend.tau_hi},
    include_null_controls=False,
    control_noise_initialization=ControlNoiseInit.ZERO
)

INITIAL_STATE = np.array([np.radians(0), 0])

pend_env = Simulator(pend, controller=pend_controller)
states, controls, time = pend_env.run(simulation_length=16, initial_state=INITIAL_STATE, controlled=True)

# Plot the results
fig, axd = plt.subplot_mosaic([['ul', 'r'], ['ll', 'r']], figsize=(9, 4), layout="constrained")

fig.suptitle("Pendulum Swing Up Response")

axd["ul"].plot(time, transform_02pi_to_negpospi(states[:, 0]), label="Theta")
axd["ul"].set(xlabel="Time (s)", ylabel="Theta [rad]")

axd["ll"].plot(time, states[:, 1], label="Angular Velocity")
axd["ll"].set(xlabel="Time (s)", ylabel="Angular Velocity [rad/s]")

axd["r"].plot(time, controls[:, 0])
axd["r"].set(xlabel="Time (s)", ylabel="Torque [Nm]")

plt.tight_layout()
plt.show()