import sys
sys.path.append("..")

import numpy as np
from utils import RK4


class BaseSystem:
    def __init__(self, integrator, measurement_noise_cov=None):
        self.integrator = integrator
        self.measurement_noise_cov = measurement_noise_cov

    def simulator(self, state, control, measurement_noise=False):
        next_state = self.integrator.step(state, control)

        if measurement_noise:
            noise = np.random.multivariate_normal(np.zeros_like(control), cov=self.measurement_noise_cov,
                                                  size=1)[0]
            next_state += noise

        return next_state


class BlockOnSlope(BaseSystem):
    def __init__(self, m, slope_angle, dt, nn_model=None, measurement_noise_stddev=None):
        self.m = m
        self.theta = slope_angle
        self.g = 9.81
        self.dt = dt
        self.nx, self.nu = (2, 1)
        self.nn_model = nn_model

        self.null_action = np.zeros(self.nu)
        self.u_lo, self.u_hi = ([-10], [10])
        self.x_lo, self.x_hi = ([0, -10], [10, 10])

        if measurement_noise_stddev is None:
            measurement_noise_stddev = np.zeros(self.nu)
        self.measurement_noise_cov = np.diag(measurement_noise_stddev ** 2)

        def dynamics(control):
            if not control:
                F = 0.0
            F = control[0]
            m, theta, g = self.m, self.theta, self.g

            def get_xdot(x):
                p, pdot = x
                pddot = g * np.sin(theta) + (F / m)

                return np.array([pdot, pddot])

            return get_xdot

        self.integrator = RK4(dynamics=dynamics, dt=self.dt)

        super().__init__(self.integrator, self.measurement_noise_cov)

    def __str__(self):
        return "pendulum"


class Pendulum(BaseSystem):
    def __init__(self, m, l, b, dt, nn_model=None, measurement_noise_stddev=None):
        self.m = m
        self.l = l
        self.b = b
        self.g = 9.81
        self.dt = dt
        self.nx, self.nu = (2, 1)
        self.nn_model = nn_model

        self.null_action = np.zeros(self.nu)
        self.u_lo, self.u_hi = ([-4], [4])
        self.x_lo, self.x_hi = ([0, -2 * np.pi], [2 * np.pi, 2 * np.pi])

        if measurement_noise_stddev is None:
            measurement_noise_stddev = np.zeros(self.nu)
        self.measurement_noise_cov = np.diag(measurement_noise_stddev ** 2)

        def dynamics(control):
            if not control:
                tau = 0.0
            tau = control[0]
            m, l, b, g = self.m, self.l, self.b, self.g

            def get_xdot(x):
                theta, thetadot = x
                thetaddot = (1 / (m * l ** 2)) * (tau - m * g * l * np.sin(theta) - b * thetadot)

                return np.array([thetadot, thetaddot])

            return get_xdot

        self.integrator = RK4(dynamics=dynamics, dt=self.dt)

        super().__init__(self.integrator, self.measurement_noise_cov)

    def __str__(self):
        return "pendulum"


class CartPole(BaseSystem):
    def __init__(self, m_pole, m_cart, l, dt, measurement_noise_stddev=None):
        self.m_pole = m_pole
        self.m_cart = m_cart
        self.l = l
        self.dt = dt
        self.g = 9.81
        self.nx, self.nu = (4, 1)

        self.null_action = np.zeros(self.nu)

        if measurement_noise_stddev is None:
            measurement_noise_stddev = np.zeros(self.nu)
        self.measurement_noise_cov = np.diag(measurement_noise_stddev ** 2)

        def dynamics(control):
            if not control:
                f_x = 0.0
            f_x = control[0]
            m_p, m_c, l, g, dt = self.m_pole, self.m_cart, self.l, self.g, self.dt

            def get_xdot(x):
                p, pdot, theta, thetadot = x

                pddot = (1 / (m_c + m_p * (np.sin(theta) ** 2))) * \
                          (f_x + m_p * np.sin(theta) * (l * (thetadot ** 2) + g * np.cos(theta)))

                thetaddot = (1 / (l * (m_c + m_p * (np.sin(theta) ** 2)))) * \
                            (-f_x * np.cos(theta) +
                             -m_p * l * (thetadot ** 2) * np.cos(theta) * np.sin(theta) +
                             -(m_c + m_p) * g * np.sin(theta))

                return np.array([pdot, pddot, thetadot, thetaddot])

            return get_xdot

        self.integrator = RK4(dynamics=dynamics, dt=self.dt)

        super().__init__(self.integrator, self.measurement_noise_cov)

    def __str__(self):
        return "cart_pole"

