import numpy as np
from time import perf_counter


def transform_angle_error(error):
    if error > np.pi:
        return 2 * np.pi - error
    elif error < -np.pi:
        return -2 * np.pi - error
    return error


def transform_02pi_to_negpospi(angles):
    return np.where(angles > np.pi, angles - 2 * np.pi, angles)


class Simulator:
    def __init__(self, ds, controller=None):
        self.ds = ds
        self.controller = controller

    def run(self, simulation_length, initial_state,
            controlled=False, measurement_noise=False, save_fname=None):

        if self.controller is None:
            controlled = False

        n_steps = int((1 / self.ds.dt) * simulation_length)

        controls, states = np.empty((n_steps, self.ds.nu)), np.empty((n_steps, self.ds.nx))

        # Begin simulation
        current_state = initial_state
        start_time = perf_counter()

        for i in range(0, n_steps):
            if not controlled:
                action = self.ds.null_action
            else:
                action = self.controller.step(current_state)

            current_state = self.ds.simulator(current_state, action,
                                              measurement_noise=measurement_noise)
            current_state = self.ds.ensure_state(current_state)

            states[i] = current_state
            controls[i] = action

        end_time = perf_counter()
        print("[Simulator] [Info] {} simulation elapsed time: {:.5f} s".format(str(self.ds), end_time - start_time))

        if save_fname: self.save(states, controls, save_fname)
        time_points = time = np.linspace(0, simulation_length, n_steps)

        return states, controls, time_points

    def save(self, states, controls, save_fname):
        np.savez(save_fname, states=states, controls=controls)


class RK4:
    def __init__(self, dynamics, dt):
        self.dynamics = dynamics
        self.dt = dt

    def step(self, state, control):
        get_xdot = self.dynamics(control)

        def rk4_update(state, dt):
            k1 = get_xdot(state)
            k2 = get_xdot(state + dt * k1/2)
            k3 = get_xdot(state + dt * k2 / 2)
            k4 = get_xdot(state + dt * k3)

            update = (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

            return state + update

        return rk4_update(state, self.dt)


class Euler:
    def __init__(self, dynamics, dt):
        self.dynamics = dynamics
        self.dt = dt

    def step(self, state, control):
        get_xdot = self.dynamics(control)
        x_dot = get_xdot(state)

        q_dot = x_dot[::2]
        q_ddot = x_dot[1::2]

        update = np.zeros_like(state, dtype=np.float64)

        update[::2] += q_dot * self.dt + 1 / 2 * q_ddot * (self.dt ** 2)
        update[1::2] += q_ddot * self.dt

        return state + update
