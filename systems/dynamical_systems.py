import sys
sys.path.append("..")

import numpy as np
from utils import RK4, Euler


class BaseSystem:
    def __init__(self, dt, integrator, measurement_noise_cov=None):
        self.dt = dt
        self.integrator = integrator
        self.measurement_noise_cov = measurement_noise_cov

    def set_integrator(self, integrator_fn): self.integrator = integrator_fn
    def ensure_state(self, x): return x
    def ensure_control(self, u): return u
    
    def kinematic_evolve_state(self, states, controls):
        x_ddots = controls
        x_dots = states[:, 1::2]

        updates = np.zeros_like(states)
        updates[:, ::2] += x_dots * self.dt + 1 / 2 * x_ddots * (self.dt ** 2)
        updates[:, 1::2] += x_ddots * self.dt

        return states + updates

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

        self.u_lo, self.u_hi = np.array([-2]), np.array([2])
        self.accel_lo, self.accel_hi = np.array([-10]), np.array([10])

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


class FK2R:
    def __init__(self, arm):
        """
        Forward kinematics for 2R planar arm
        :param arm: PlanarArm object
        """
        self.arm = arm

    def solve(self, t1, t2):
        l1, l2 = self.arm.l1, self.arm.l2

        x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
        y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
        theta = t1 + t2

        return np.array([x, y, theta])


class IK2R:
    def __init__(self, arm):
        """
        Inverse Kinematics for 2R planar arm
        :param arm: PlanarArm object
        """
        self.arm = arm

    @staticmethod
    def slack_eq(slack):
        def eq(a, b):
            return (a - slack <= b <= a + slack)

        return eq

    def solve(self, desired_pose):
        l1, l2 = self.arm.l1, self.arm.l2
        desired_position = desired_pose[:2]
        x, y = desired_position

        eq = IK2R.slack_eq(1e-5)
        c2 = (np.linalg.norm(desired_position) ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)

        if abs(c2) > 1 + 1e-5:
            return None
        elif eq(c2, 1):
            return np.array([np.arctan2(y, x), 0])
        elif eq(c2, -1) and np.any(desired_position):
            return np.array([np.arctan2(y, x), np.pi])
        elif eq(c2, -1) and not np.any(desired_position):
            return np.array([np.pi, np.pi])

        candidate_theta2s = [np.arccos(c2), -np.arccos(c2)]
        candidate_theta1s = []

        eef_theta = np.arctan2(y, x)

        for k in [0, 1]:
            theta_2 = candidate_theta2s[k]
            candidate_theta1s.append(
                eef_theta - np.arctan2(l2 * np.sin(theta_2), l1 + l2 * np.cos(theta_2))
            )

        candidate_thetas = list(zip(candidate_theta1s, candidate_theta2s))

        return np.array(candidate_thetas[0])


class Planar2RArmBase(BaseSystem):
    def __init__(self, m1, m2, l1, l2, r1, r2, I1, I2, dt,
                 measurement_noise_stddev=None):
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.r1, self.r2 = r1, r2
        self.I1, self.I2 = I1, I2
        self.g = 9.81
        self.dt = dt
        self.nx, self.nu = (2 * 2, 2)

        self.null_action = np.zeros(self.nu)

        self.u_lo, self.u_hi = np.array([-10, -10]), np.array([10, 10])
        self.accel_lo, self.accel_hi = np.array([-4,0, -4.0]), np.array([4.0, 4.0])

        self.x_lo, self.x_hi = np.array([0, -4, 0, -4]), np.array([2*np.pi, 4, 2*np.pi, 4])

        if measurement_noise_stddev is None:
            measurement_noise_stddev = np.zeros(self.nu)
        self.measurement_noise_cov = np.diag(measurement_noise_stddev ** 2)

        def M_mat(x):
            t1, dot_t1, t2, dot_t2 = x
            c2 = np.cos(t2)

            M = np.array([
                [(m1 + m2) * l1 ** 2 + m2 * l2 ** 2 + 2 * m2 * l1 * l2 * c2, m2 * l2 ** 2 + m2 * l1 * l1 * c2],
                [m2 * l2 ** 2 + m2 * l1 * l2 * c2, m2 * l2 ** 2]
            ])

            return M

        def C_mat(x):
            t1, dot_t1, t2, dot_t2 = x
            s2 = np.sin(t2)

            C = np.array([
                [0, -m2 * l1 * l2 * (2 * dot_t1 + dot_t2) * s2],
                [0.5 * m2 * l1 * l2 * (2 * dot_t1 + dot_t2) * s2, -0.5 * m2 * l1 * l2 * dot_t1 * s2]
            ])

            return C

        def N_mat(x):
            t1, dot_t1, t2, dot_t2 = x
            s1 = np.sin(t1)
            s12 = np.sin(t1 + t2)
            g = self.g

            N = -g * np.array([
                (m1 + m2) * l1 * s1 + m2 * l2 * s12,
                m2 * l2 * s12
            ])

            return N

        self.M_mat, self.C_mat, self.N_mat = M_mat, C_mat, N_mat

        def dynamics(control):
            tau = control
            if control is None:
                tau = np.zeros(self.nu)

            def get_dotx(x):
                t1, dot_t1, t2, dot_t2 = x

                M, C, N = self.M_mat(x), self.C_mat(x), self.N_mat(x)

                # Manipulator equation is of the form:
                #   M(q) ddot(q)  + C(q, dot(q)) dot(q) = N + Bu

                dot_t = x[1:4:2]
                ddot_t = np.dot(np.linalg.inv(M), (tau + N - np.dot(C, dot_t)))

                return np.array([dot_t1, ddot_t[0], dot_t2, ddot_t[1]])

            return get_dotx

        self.integrator = RK4(dynamics=dynamics, dt=self.dt)

        super().__init__(self.dt, self.integrator, self.measurement_noise_cov)

    def fk(self, t1, t2):
        t1 = t1 - np.pi/2
        l1, l2 = self.l1, self.l2

        x = l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
        y = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
        theta = t1 + t2

        return np.array([x, y, theta])

    def inverse_dynamics(self, x, ddot_t):
        M, C, N = self.M_mat(x), self.C_mat(x), self.N_mat(x)
        dot_t = x[1:4:2]

        # Manipulator equation is of the form:
        #   M(q) ddot(q)  + C(q, dot(q)) dot(q) = N + Bu

        tau = np.dot(M, ddot_t) + np.dot(C, dot_t) - N
        return tau

    def ensure_state(self, x):
        t1, dot_t1, t2, dot_t2 = x
        tau = 2*np.pi

        return np.array([t1 % tau, dot_t1, t2 % tau, dot_t2])

    def ensure_control(self, tau):
        return np.clip(tau, self.u_lo, self.u_hi)

    def __str__(self):
        return "planar_2r_arm"


class Planar2RArm(Planar2RArmBase):
    def __init__(self, m1, m2, l1, l2, r1, r2, I1, I2, dt, measurement_noise_stddev=None):
        super().__init__(m1, m2, l1, l2, r1, r2, I1, I2, dt, measurement_noise_stddev)
        self.nx, self.nu = (3 * 2, 2)

        def dynamics(control):
            tau = control
            if control is None:
                tau = np.zeros(self.nu)

            def get_dotx(x):
                cleaned_x = x[:4]
                t1, dot_t1, t2, dot_t2, param, dot_param = x

                M, C, N = self.M_mat(cleaned_x), self.C_mat(cleaned_x), self.N_mat(cleaned_x)

                # Manipulator equation is of the form:
                #   M(q) ddot(q)  + C(q, dot(q)) dot(q) = N + Bu

                dot_t = cleaned_x[1::2]
                ddot_t = np.dot(np.linalg.inv(M), (tau + N - np.dot(C, dot_t)))

                return np.array([dot_t1, ddot_t[0], dot_t2, ddot_t[1], dot_param, 0])

            return get_dotx

        self.integrator = RK4(dynamics=dynamics, dt=self.dt)
        self.set_integrator(self.integrator)

    def inverse_dynamics(self, x, ddot_t):
        return super().inverse_dynamics(x[:4], ddot_t)

    def ensure_state(self, x):
        t1, dot_t1, t2, dot_t2, param, dot_param = x
        tau = 2*np.pi

        return np.array([t1 % tau, dot_t1, t2 % tau, dot_t2, param, dot_param])

    def kinematic_evolve_state(self, states, controls):
        n = controls.shape[0]
        controls = np.hstack((controls, np.zeros(n).reshape((-1, 1))))
        return super().kinematic_evolve_state(states, controls)

    def __str__(self):
        return "planar_2r_arm"
