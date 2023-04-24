import numpy as np


def transform_angle_error(error):
    if error > np.pi:
        return 2 * np.pi - error
    elif error < -np.pi:
        return -2 * np.pi - error
    return error


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
