from dynamical_systems import Planar2RArm, FK2R, IK2R, Pendulum
import sys
sys.path.append("..")

from control.system_controller import SystemController, ControlNoiseInit
from utils import transform_angle_error, transform_02pi_to_negpospi, Simulator, Path, Trajectory

import numpy as np
import time
import matplotlib.pyplot as plt
from time import perf_counter

def test_pend():
    from systems.dynamical_systems import Pendulum

    pend = Pendulum(m=1, l=1, b=0.1, dt=1 / 10)

    DESIRED_THETA = np.pi
    DESIRED_OMEGA = 0.0

    def terminal_cost(x):
        theta, omega = x

        theta_error = np.cos(theta - DESIRED_THETA + np.pi) + 1
        omega_error = DESIRED_OMEGA - omega

        error = np.array([theta_error, omega_error])
        Q = np.diag([10, 1])

        return error.T @ Q @ error

    def state_cost(x):
        theta, omega = x

        theta_error = np.cos(theta - DESIRED_THETA + np.pi) + 1

        error = np.array([theta_error])
        Q = np.diag([1])

        return error.T @ Q @ error

    pend_controller = SystemController(
        ds=pend,
        n_rollouts=100,
        horizon_length=10,
        exploration_cov=np.diag([0.5 ** 2]),
        exploration_lambda=1e-2,
        alpha_u=0.9,
        state_cost=state_cost, terminal_cost=terminal_cost,
        control_range={"min": pend.tau_lo, "max": pend.tau_hi},
        include_null_controls=False
    )

    INITIAL_STATE = np.array([np.radians(0), 0])

    pend_env = Simulator(pend, controller=pend_controller)
    states, controls, time = pend_env.run(simulation_length=12, initial_state=INITIAL_STATE, controlled=True)

    # Plot the results
    fig, axd = plt.subplot_mosaic([['ul', 'r'], ['ll', 'r']], figsize=(9, 4), layout="constrained")

    fig.suptitle("Pendulum Swing Up Response")

    axd["ul"].plot(time, states[:, 0], label="Theta")
    axd["ul"].set(xlabel="Time (s)", ylabel="Theta [rad]")

    axd["ll"].plot(time, states[:, 1], label="Angular Velocity")
    axd["ll"].set(xlabel="Time (s)", ylabel="Angular Velocity [rad/s]")

    axd["r"].plot(time, controls[:, 0])
    axd["r"].set(xlabel="Time (s)", ylabel="Torque [Nm]")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_pend()
    """M1 = M2 = 1
    L1 = L2 = 1
    R1 = R2 = L1 / 2
    I1 = I2 = 1/3 * M1 * L1**2
    DT = 1/20

    INITIAL_STATE = np.array([np.radians(90), 0, np.radians(0), 0,
                              0, 1])

    arm = Planar2RArm(
        m1=M1, m2=M2, l1=L1, l2=L2, r1=R1, r2=R2, I1=I1, I2=I2, dt=DT
    )

    print(arm.IK.solve(desired_pose=np.array([1, 0, np.pi])))
    print(FK.solve(*IK.solve(desired_pose=np.array([1, 0, np.pi]))))

    # next_state = arm.integrator.step(INITIAL_STATE, arm.null_action)
    # ivd = arm.inverse_dynamics(x=np.array([np.pi/2, 0, 0, 0, 0, 1]), ddot_t=np.array([0, 0]))
    # print(ivd)
    # arm_env = Simulator(arm, controller=None)
    # states, controls, time = arm_env.run(simulation_length=5, initial_state=INITIAL_STATE,
    #                                      control_action=np.array([29.42, 9.81]))
    # np.set_printoptions(precision=5, suppress=True)
    #
    # import matplotlib.pyplot as plt
    # positions = np.empty((len(states), 3))
    # for (i, (t1, t1dot, t2, t2dot, param, _)) in enumerate(states):
    #     positions[i] = arm.forward_kinematics(t1, t2)
    # print(positions)
    # plt.plot(positions[:, 0], positions[:, 1])
    # plt.xlim(-2.2, 2.2)
    # plt.ylim(-2.2, 2.2)
    # plt.show()"""


