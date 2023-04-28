import mppi_controller
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter


sns.set_style("whitegrid")
sns.set_palette("tab10")


class InvertedPendulum:
    def __init__(self, initial_state, m, l, g, dt):
        self.initial_state = initial_state
        self.desired_state = None
        self.m = m
        self.l = l
        self.g = g
        self.dt = dt
        self.nx, self.nu = (2, 1)

        self.controller = mppi_controller.MPPIController(
            n_rollouts=200,
            horizon_length=10,
            exploration_cov=np.diag([1.5 ** 2]),
            exploration_lambda=1e-3,
            nx=self.nx,
            nu=self.nu,
            terminal_cost=self.terminal_cost,
            state_cost=self.state_cost,
            evolve_state=self.evolve_state,
            dt=dt,
            control_range={"min": np.array([-15]), "max": np.array([15])},
            control_noise_initialization=mppi_controller.ControlNoiseInit.LAST
        )

        self.measurement_noise_cov = np.array([
            [(3 * np.pi / 180) ** 2]  # 5 degrees of measurement noise
        ])

    def terminal_cost(self, x):
        error = (self.desired_state - x)
        Q = np.diag([1, 1])

        return error.T @ Q @ error

    def state_cost(self, x):
        error = (self.desired_state - x)
        Q = np.diag([3, 1])

        return error.T @ Q @ error

    def evolve_state(self, x, u, dt):
        b = 1
        th, thdot = x
        thddot = (-b * thdot + u[0] + self.m * self.g * self.l * np.sin(th)) / (self.m * self.l ** 2)

        newthdot = thdot + thddot * dt
        newth = th + newthdot * dt + (0.5 * thddot * dt ** 2)

        evolved_state = np.array([newth, newthdot])

        return evolved_state

    def simulator(self, x, u, measurement_noise=False):
        curr_state = self.evolve_state(x, u, dt=self.dt)
        if measurement_noise:
            noise = np.random.multivariate_normal(np.zeros_like(u), cov=self.measurement_noise_cov,
                                                  size=1)[0]
            curr_state += noise

        return curr_state

dt = 1 / 10

inv_pend_env = InvertedPendulum(
    initial_state=np.array([0, 0]),
    m=1,     # kg
    l=1,     # m
    g=-9.81, # m / s^2
    dt=dt    # s
)

inv_pend_env.desired_state = np.array([np.pi, 0])

# Create lists to store control sequence and state sequences for MPPI runs
control_seq, states = [], []

simulation_length = 5  # s
n_steps = int((1 / dt) * simulation_length)

# Begin simulation
current_state = inv_pend_env.initial_state

for _ in range(0, n_steps):
    action = inv_pend_env.controller.step(current_state)
    current_state = inv_pend_env.simulator(current_state, action, measurement_noise=False)

    states.append(current_state[0])
    control_seq.append(action)