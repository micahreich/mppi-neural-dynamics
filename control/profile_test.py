import sys
sys.path.append("..")

from system_controller import SystemController, ControlNoiseInit
from utils import transform_angle_error, transform_02pi_to_negpospi, Simulator, Path, Trajectory

import numpy as np
import time
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.integrate import odeint

plt.style.use(["science", "grid", "vibrant"])
ls = {"loose dash": (0, (5, 10)), "dash": (0, (5, 5))}

from systems.dynamical_systems import Planar2RArm, FK2R, IK2R

M1 = M2 = 1
L1 = L2 = 1
R1 = R2 = L1 / 2
I1 = I2 = 1/3 * M1 * L1**2

arm = Planar2RArm(
    m1=M1, m2=M2, l1=L1, l2=L2, r1=R1, r2=R2, I1=I1, I2=I2, dt=1/20
)

ik = IK2R(arm)

def line(t):
    L=1.5
    return np.array([L*(1-t) + 2 - L, 0.0])

def semicircle(t):
    R = 2
    return np.array([R*np.cos(np.pi*t)+(2-R), R*np.sin(np.pi*t)])

def circle(t):
    R, tau = 0.8, 2*np.pi
    return np.array([R*np.cos(tau*t)+(2-R), R*np.sin(tau*t)])

def infty(t):
    R, tau = 2, 2*np.pi
    return np.array([R*np.cos(tau*t), R*np.sin(tau*t)*np.cos(tau*t)])

path = Path(parametric_eqn=infty, time_length=8)

def joint_semicircle(t):
    return np.array([np.pi*t + np.pi/2, 0])

def joint_rangle(t):
    return np.array([np.pi/2 * t + np.pi/2, np.pi/2 * t])

joint_path = Path(parametric_eqn=joint_semicircle, time_length=4)

ACCEL_STDDEV = 3.0

# Define the controller cost functions
def terminal_cost(path):
    def tc(x):
        t1, dot_t1, t2, dot_t2, param, _ = x
        DESIRED_POSITION = path(param)
    
        pose = arm.fk(t1, t2)
        pos_error = np.linalg.norm(DESIRED_POSITION - pose[:2])

        error = np.array([pos_error, dot_t1, dot_t2])
        Q = np.diag([5e2, 5e2, 5e1])

        # DESIRED_T1, DESIRED_T2 = joint_path(param)
        # t1_error = np.cos(t1 - DESIRED_T1 + np.pi) + 1
        # t2_error = np.cos(t2 - DESIRED_T2 + np.pi) + 1
        
        # error = np.array([t1_error, t2_error])
        # Q = np.diag([2e4, 2e4])

        return np.dot(error, np.dot(Q, error))
    return tc

def state_cost(path):
    def sc(x):
        t1, dot_t1, t2, dot_t2, param, _ = x
        DESIRED_POSITION = path(param)

        pose = arm.fk(t1, t2)
        pos_error = np.linalg.norm(DESIRED_POSITION - pose[:2])

        error = np.array([pos_error])
        Q = np.diag([5e3])

        # DESIRED_T1, DESIRED_T2 = joint_path(param)
        # t1_error = np.cos(t1 - DESIRED_T1 + np.pi) + 1
        # t2_error = np.cos(t2 - DESIRED_T2 + np.pi) + 1
        
        # error = np.array([t1_error, t2_error])
        # Q = np.diag([2e4, 2e4])

        return np.dot(error, np.dot(Q, error))
    return sc

arm.u_lo = np.array([-6.0, -6.0])
arm.u_hi = -arm.u_lo

arm_controller = SystemController(
    ds=arm,
    inverse_dyn_control=True,
    n_rollouts=150,
    horizon_length=30,
    exploration_cov=np.diag([ACCEL_STDDEV ** 2, ACCEL_STDDEV ** 2]),
    exploration_lambda=1e-4,
    alpha_mu=0.2,
    alpha_sigma=0.99,
    terminal_cost=terminal_cost(path),
    state_cost=state_cost(path),
    control_range={"min": arm.u_lo, "max": arm.u_hi},
    include_null_controls=False,
    control_noise_initialization=ControlNoiseInit.ZERO
)

T1_0, T2_0 = ik.solve(np.array([2, 0, 0])) + np.array([np.pi/2, 0])
INITIAL_STATE = np.array([T1_0, 0, T2_0, 0, 0, 1])

def main():
    arm_env = Simulator(arm, controller=arm_controller)
    states, controls, times = arm_env.run(simulation_length=4, initial_state=INITIAL_STATE, controlled=True)

if __name__ == '__main__':
    import cProfile, pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime', 'cumtime')
    stats.print_stats()