import sys
sys.path.append("..")

import numpy as np
from utils import RK4, Euler


class BaseSystem:
    def __init__(self, dt, integrator, measurement_noise_cov=None):
        self.dt = dt
        self.integrator = integrator
        self.measurement_noise_cov = measurement_noise_cov

    def ensure_state(self, x): return x
    def ensure_control(self, u): return u

    def kinematic_evolve_state(self, states, controls):
        x_ddots = controls
        x_dots = states[:, 1::2]

        next_states = states
        next_states[:, ::2] += x_dots * self.dt + 1 / 2 * x_ddots * (self.dt ** 2)
        next_states[:, 1::2] += x_ddots * self.dt

        return next_states

    def simulator(self, state, control, measurement_noise=False):
        next_state = self.integrator.step(state, control)

        if measurement_noise:
            noise = np.random.multivariate_normal(np.zeros_like(control), cov=self.measurement_noise_cov,
                                                  size=1)[0]
            next_state += noise

        return next_state


class BlockOnSlope(BaseSystem):
    def __init__(self, m, slope_angle, dt, measurement_noise_stddev=None):
        self.m = m
        self.theta = slope_angle
        self.g = 9.81
        self.dt = dt
        self.nx, self.nu = (2, 1)

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

            def get_dotx(x):
                p, dot_p = x
                ddot_p = g * np.sin(theta) + (F / m)

                return np.array([dot_p, ddot_p])

            return get_dotx

        self.integrator = RK4(dynamics=dynamics, dt=self.dt)

        super().__init__(self.integrator, self.measurement_noise_cov)

    def __str__(self):
        return "block_on_slope"


class Pendulum(BaseSystem):
    def __init__(self, m, l, b, dt, measurement_noise_stddev=None):
        self.m = m
        self.l = l
        self.b = b
        self.g = 9.81
        self.dt = dt
        self.nx, self.nu = (2, 1)

        self.null_action = np.zeros(self.nu)

        self.tau_lo, self.tau_hi = np.array([-2]), np.array([2])
        self.alpha_lo, self.alpha_hi = np.array([-10]), np.array([10])

        self.x_lo, self.x_hi = np.array([0, -2 * np.pi]), np.array([2 * np.pi, 2 * np.pi])

        if measurement_noise_stddev is None:
            measurement_noise_stddev = np.zeros(self.nu)
        self.measurement_noise_cov = np.diag(measurement_noise_stddev ** 2)

        def dynamics(control):
            if not control:
                tau = 0.0
            tau = control[0]
            m, l, b, g = self.m, self.l, self.b, self.g

            def get_dotx(x):
                theta, dot_theta = x
                ddot_theta = (1 / (m * l ** 2)) * (tau - m * g * l * np.sin(theta) - b * dot_theta)

                return np.array([dot_theta, ddot_theta])

            return get_dotx

        self.integrator = RK4(dynamics=dynamics, dt=self.dt)

        super().__init__(self.dt, self.integrator, self.measurement_noise_cov)

    def inverse_dynamics(self, theta_ddot, state):
        m, l, b, g = self.m, self.l, self.b, self.g
        theta, dot_theta = state

        return theta_ddot * m * l**2 + b * dot_theta + m * g * l * np.sin(theta)

    def ensure_state(self, x):
        theta, dot_theta = x
        tau = 2 * np.pi

        return np.array([theta % tau, dot_theta])

    def ensure_control(self, tau):
        return np.clip(tau, self.tau_lo, self.tau_hi)

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
        self.u_lo, self.u_hi = np.array([-10]), np.array([10])
        self.x_lo, self.x_hi = np.array([-100, -10, 0, -10]), np.array([100, 10, 2*np.pi, 10])

        if measurement_noise_stddev is None:
            measurement_noise_stddev = np.zeros(self.nu)
        self.measurement_noise_cov = np.diag(measurement_noise_stddev ** 2)

        def dynamics(control):
            if not control:
                f_x = 0.0
            f_x = control[0]
            m_p, m_c, l, g, dt = self.m_pole, self.m_cart, self.l, self.g, self.dt

            def get_dotx(x):
                p, dot_p, theta, dot_theta = x

                ddot_p = (1 / (m_c + m_p * (np.sin(theta) ** 2))) * \
                          (f_x + m_p * np.sin(theta) * (l * (dot_theta ** 2) + g * np.cos(theta)))

                ddot_theta = (1 / (l * (m_c + m_p * (np.sin(theta) ** 2)))) * \
                            (-f_x * np.cos(theta) +
                             -m_p * l * (dot_theta ** 2) * np.cos(theta) * np.sin(theta) +
                             -(m_c + m_p) * g * np.sin(theta))

                return np.array([dot_p, ddot_p, dot_theta, ddot_theta])

            return get_dotx

        self.integrator = RK4(dynamics=dynamics, dt=self.dt)

        super().__init__(self.dt, self.integrator, self.measurement_noise_cov)

    def ensure_state(self, x):
        p, dot_p, theta, dot_theta = x
        tau = 2 * np.pi

        return np.array([p, dot_p, theta % tau, dot_theta])

    def __str__(self):
        return "cart_pole"


class Planar2RArm(BaseSystem):
    def __init__(self, m1, m2, l1, l2, r1, r2, I1, I2, dt, measurement_noise_stddev=None):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.r1, self.r2 = r1, r2
        self.I1, self.I2 = I1, I2
        self.g = 9.81
        self.dt = dt
        self.nx, self.nu = (2 * 2, 2)

        self.null_action = np.zeros(self.nu)

        self.tau_lo, self.tau_hi = np.array([-10]), np.array([10])
        self.alpha_lo, self.alpha_hi = np.array([-10]), np.array([10])

        self.x_lo, self.x_hi = np.array([0, -10, 0, -10]), np.array([2*np.pi, 10, 2*np.pi, 10])
        self.u_lo, self.u_hi = np.array([-10, -10]), np.array([10, 10])

        if measurement_noise_stddev is None:
            measurement_noise_stddev = np.zeros(self.nu)
        self.measurement_noise_cov = np.diag(measurement_noise_stddev ** 2)

        def dynamics(control):
            tau = control

            if control is None:
                tau = np.zeros(self.nu)

            tau1, tau2 = tau
            m1, m2, l1, l2, r1, r2, I1, I2, g = self.m1, self.m2, self.l1, self.l2, self.r1, self.r2, self.I1, self.I2, self.g

            def get_dotx(x):
                t1, dot_t1, t2, dot_t2 = x

                # Generalized mass matrix -> M(\theta)
                M11 = I1 + I2 + m1 * (r1 ** 2) + m2 * (l1 ** 2 + r2 ** 2 + 2 * l1 * r2 * np.cos(t2))
                M12 = I2 + m2 * (r2 ** 2 + l1 * r2 * np.cos(t2))
                M21 = I2 + m2 * (r2 ** 2 + l1 * r2 * np.cos(t2))
                M22 = I2 + m2 * (r2 ** 2)

                # Coriolis and centrifugal terms -> b(\theta, \dot\theta)
                b1 = -m2 * l1 * r2 * np.sin(t2) * (2 * t1 * t2 + dot_t2 ** 2)
                b2 = m2 * l1 * r2 * np.sin(t2) * dot_t1 ** 2
                b_Mat = np.array([[b1], [b2]])

                # Gravitational terms -> g(\theta)
                g1 = m1 * g * r1 * np.cos(t1) + m2 * g * (l1 * np.cos(t1) + r2 * np.cos(t1 + t2))
                g2 = m2 * g * r2 * np.cos(t1 + t2)
                g_Mat = np.array([[g1], [g2]])

                # \tau -> External generalized forces.
                tau_Mat = np.array([[tau1], [tau2]])

                M_Mat = np.array([[M11, M12], [M21, M22]])
                t_ddots = np.linalg.inv(M_Mat).dot(-b_Mat - g_Mat) + tau_Mat

                return np.array([dot_t1, t_ddots[0][0], dot_t2, t_ddots[1][0]])

            return get_dotx

        self.integrator = Euler(dynamics=dynamics, dt=self.dt)

        super().__init__(self.dt, self.integrator, self.measurement_noise_cov)

    def forward_kinematics(self, t1, t2):
        l1, l2 = self.l1, self.l2

        x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
        y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
        theta = t1 + t2

        return np.array([x, y, theta])

    def ensure_state(self, x):
        t1, t1_dot, t2, t2_dot = x
        tau = 2*np.pi

        return np.array([t1 % tau, t1_dot, t2 % tau, t2_dot])

    def ensure_control(self, tau):
        return np.clip(tau, self.tau_lo, self.tau_hi)

    def __str__(self):
        return "planar_2r_arm"
