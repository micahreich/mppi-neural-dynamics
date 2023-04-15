import scipy
import numpy as np


class InvertedPendulum:
    def __init__(self, m, l, b, g, dt):
        self.m = m
        self.l = l
        self.b = b
        self.g = g
        self.dt = dt
        self.nx, self.nu = (2, 1)

        self.u_lo, self.u_hi = ([-10], [10])
        self.x_lo, self.x_hi = ([0, -2 * np.pi], [2 * np.pi, 2 * np.pi])

    def dynamics(self, u):
        def dydt(y, t):
            theta, omega = y
            alpha = (-self.b * omega + u[0] + self.m * self.g * self.l * np.sin(theta)) / (self.m * self.l**2)

            return np.array([omega, alpha])

        return dydt

    def __str__(self):
        return "inverted_pendulum"
