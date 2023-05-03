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
            controlled=False, measurement_noise=False, save_fname=None,
            control_action=None, debug=False, timing=False):

        if self.controller is None:
            controlled = False

        n_steps = int((1 / self.ds.dt) * simulation_length)

        controls, states = np.empty((n_steps, self.ds.nu)), np.empty((n_steps, self.ds.nx))

        # Begin simulation
        current_state = initial_state
        start_time = perf_counter()

        for i in range(0, n_steps):
            if control_action is not None:
                action = control_action
            elif not controlled:
                action = self.ds.null_action
            else:
                action = self.controller.step(current_state, debug=debug)

            current_state = self.ds.simulator(current_state, action,
                                              measurement_noise=measurement_noise)
            current_state = self.ds.ensure_state(current_state)

            states[i] = current_state
            controls[i] = action

        end_time = perf_counter()
        
        if timing:
            print("[Simulator] [Info] {} simulation elapsed time: {:.5f} s".format(str(self.ds), end_time - start_time))

        if save_fname: self.save(states, controls, save_fname)
        times = np.linspace(0, simulation_length, n_steps)

        return states, controls, times

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


class ParametricPath:
    def __init__(self, parametric_eqn, time_length):
        """
        General path class as defined by a parametric curve which takes inputs from [0, 1] and
        returns the output in the workspace.

        :param parametric_eqn: function: (float) -> (nx)
        :param time_length: The amount of time in seconds over which the path should be followed
        """
        self.parametric_eqn = parametric_eqn
        self.time_length = time_length

    def get_points(self, times):
        n_pts = len(times)
        point_shape = self.parametric_eqn(0).shape[0]

        points = np.empty(shape=(n_pts, point_shape))

        for (i, t) in enumerate(times):
            points[i] = self.__call__(t)

        return points
    
    def __call__(self, t):
        if t > self.time_length:
            return self.parametric_eqn(1)
        return self.parametric_eqn(t / self.time_length)


class Trajectory(ParametricPath):
    def __init__(self, position_parametric_eqn, velocity_parametric_eqn, time_length):
        self.position_parametric_eqn = position_parametric_eqn
        self.velocity_parametric_eqn = velocity_parametric_eqn
        self.time_length = self.time_length

        super().__init__(position_parametric_eqn, time_length)

    def trapezoidal_velocity_profile(self, v_max, a_max):
        risen_time = v_max / a_max

        def velocity_parametric_eqn(t):
            if 0 <= t <= risen_time:
                return a_max * t
            elif risen_time < t < self.time_length - risen_time:
                return v_max
            elif self.time_length - risen_time <= t <= self.time_length:
                return -a_max * (t - self.time_length - risen_time) - v_max
            else:
                return 0

        return velocity_parametric_eqn
