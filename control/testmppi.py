import MPPIController
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter


sns.set_style("whitegrid")
sns.set_palette("tab10")


class BlockOnSlope:
    # Apply a force parallel to the slope to achieve a desired position
    # and velocity of the block

    def __init__(self, initial_state, block_mass, slope_angle, g, dt, mu_k=0):
        self.initial_state = initial_state
        self.desired_state = None
        self.block_mass = block_mass
        self.slope_angle = slope_angle
        self.g = g
        self.dt = dt
        self.mu_k = mu_k
        self.nx, self.nu = (2, 1)

        self.controller = MPPIController.MPPIController(
            n_rollouts=200,
            horizon_length=10,
            exploration_cov=np.diag([10 ** 2]),
            exploration_lambda=1e-2,
            nx=self.nx,
            nu=self.nu,
            terminal_cost=self.terminal_cost,
            state_cost=self.state_cost,
            evolve_state=self.evolve_state,
            dt=dt
        )

    def terminal_cost(self, x):
        error = (self.desired_state - x)
        Q = np.array([[3, 0], [0, 1]])

        return error.T @ Q @ error

    def state_cost(self, x):
        error = (self.desired_state - x)
        Q = np.array([[1, 0], [0, 1]])

        return error.T @ Q @ error

    def evolve_state(self, x, u, dt):
        pos, vel = x
        accel = (self.g * np.sin(self.slope_angle)) + (u[0] / self.block_mass)

        newvel = vel + accel * dt
        newpos = pos + newvel * dt + 0.5 * accel * (dt ** 2)

        evolved_state = np.array([newpos, newvel])

        return evolved_state

    def simulator(self, x, u):
        return self.evolve_state(x, u, dt=dt)


dt = 1 / 10

block_env = BlockOnSlope(
    initial_state=np.array([0.0, 0.0]),
    block_mass=5,               # kg
    slope_angle=np.radians(30), # radians
    g=-9.81,                     # m / s^2
    dt=dt                       # s
)

block_env.desired_state = np.array([10, 0])

# Create lists to store control sequence and state sequences for MPPI runs
control_seq, states = [], []

simulation_length = 10  # s
n_steps = int((1 / dt) * simulation_length)

# Begin simulation
current_state = block_env.initial_state
start_time = perf_counter()

for _ in range(0, n_steps):
    action = block_env.controller.step(current_state)
    current_state = block_env.simulator(current_state, action)

    states.append(current_state[0])
    control_seq.append(action)

print("Elapsed Time: {:.5f} s".format(perf_counter() - start_time))