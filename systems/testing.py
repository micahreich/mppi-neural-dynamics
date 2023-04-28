from dynamical_systems import Planar2RArm
from utils import Simulator
import numpy as np

if __name__ == "__main__":
    M1 = M2 = 2.25
    L1 = L2 = 1
    R1 = R2 = L1 / 2
    I1 = I2 = 1/3 * M1 * L1**2
    INITIAL_STATE = np.array([np.radians(-50), 0, np.radians(0), 0])

    arm = Planar2RArm(
        m1=M1, m2=M2, l1=L1, l2=L2, r1=R1, r2=R2, I1=I1, I2=I2, dt=1/50
    )

    arm_env = Simulator(arm, controller=None)
    states, controls, time = arm_env.run(simulation_length=10, initial_state=INITIAL_STATE)

    import matplotlib.pyplot as plt
    positions = np.empty((len(states), 3))

    for (i, (t1, t1dot, t2, t2dot)) in enumerate(states):
        positions[i] = arm.forward_kinematics(t1, t2)
    print(positions[0], positions[-1])
    plt.plot(positions[:, 0], positions[:, 1])
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


