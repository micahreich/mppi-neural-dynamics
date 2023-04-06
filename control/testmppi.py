import MPPIController
import numpy as np
import time
import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')


def block_on_slope(block_mass, slope_angle, gravity_accel, desired_state, dt):
    # Apply a force parallel to the slope to achieve a desired position
    # and velocity of the block

    nx, nu = 2, 1

    def terminal_cost(x):
        error = (desired_state - x)
        Q = np.array([[3, 0], [0, 1]])
        return error.T @ Q @ error

    def state_cost(x):
        error = (desired_state - x)
        Q = np.array([[1, 0], [0, 1]])
        return error.T @ Q @ error

    def control_cost(u, noise):
        noisy_control = u + noise
        R = np.diag([0.5 * 1e-2])

        return noisy_control.T @ R @ noisy_control

    def evolve_state(x, u, dt):
        pos, vel = x

        accel = (gravity_accel * np.sin(slope_angle)) + (u[0] / block_mass)

        evolved_state = np.array([
            pos + vel * dt + 0.5 * accel * (dt ** 2),
            vel + accel * dt
        ])

        return evolved_state

    def simulator(x, u):
        return evolve_state(x, u, dt=dt)

    controller = MPPIController.MPPIController(
        n_rollouts=200,
        horizon_length=10,
        exploration_cov=np.array([[10 ** 2]]),
        exploration_lambda=5,
        nx=nx,
        nu=nu,
        terminal_cost=terminal_cost,
        state_cost=state_cost,
        control_cost=control_cost,
        control_cov=np.array([[1]]),
        evolve_state=evolve_state,
        dt=dt
    )

    return controller, simulator

# Create the block on slope controller

sample_hz = 10
dt = 1 / sample_hz
desired_state = np.array([10, 0])

controller, simulator = block_on_slope(
    block_mass = 5, # kg
    slope_angle = np.radians(30), # radians
    gravity_accel = -9.81, # m / s^2
    desired_state = desired_state,
    dt = dt
)

# Create the control loop to run MPPI

current_state = np.array([0, 0])  # [position, velocity]
simulation_length = 15  # seconds

# Create some lists to hold plotted information
control_seq = []
xs = []

n_steps = int((1 / dt) * simulation_length)

for i in range(0, n_steps):
    next_u = controller.step(current_state)
    control_seq.append(next_u[0])

    current_state = simulator(current_state, next_u)
    xs.append(current_state[0])

# Plot the results

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle("Block on Slope Response")

time = np.linspace(0, simulation_length, n_steps)

ax1.plot(time, xs, label="block position")
ax1.plot(time, np.repeat(desired_state[0], n_steps), label="desired position")
ax1.set(xlabel="time (s)", ylabel="position (m)")
ax1.legend()

ax2.plot(time, control_seq, label="control")
ax2.set(xlabel="time (s)", ylabel="force (N)")

plt.show()