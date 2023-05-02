import sys
sys.path.append("..")

from neural_networks.dataset_generator import DatasetGenerator
from neural_networks.dynamics_networks import DynamicsNN
from systems.dynamical_systems import Planar2RArmBase
import os
import numpy as np

M1 = M2 = 1
L1 = L2 = 1
R1 = R2 = L1 / 2
I1 = I2 = 1/3 * M1 * L1**2

arm = Planar2RArmBase(
    m1=M1, m2=M2, l1=L1, l2=L2, r1=R1, r2=R2, I1=I1, I2=I2, dt=1/20
)
file_dir = os.path.dirname(os.path.realpath(__file__))


def generate_dataset(n_samples, sample_method, horizon_length, train_percentage, name, inverse_dynamics):
    SAVE_DIR = file_dir + "/data"

    # Create inverted pendulum class and dataset generator
    ds = DatasetGenerator(ds=arm, save_dir=SAVE_DIR,
                          sample_method=sample_method, inverse_dynamics=inverse_dynamics)

    # Generate data points and save to .npz file
    dataset_fname = ds.generate_dataset(n_samples=n_samples, horizon_length=horizon_length,
                                        train_percentage=train_percentage, name=name)

    # Inspect saved data
    with ds.inspect_datasets() as ds:
        print("Training dataset size: {}     Testing dataset size: {}".format(len(ds["x_train"]), len(ds["x_test"])))

    return dataset_fname


if __name__ == "__main__":
    # _ = generate_dataset(n_samples=500_000, sample_method="random rollouts",
    #                      horizon_length=20, train_percentage=0.8, name="neg4to4_500k_rollouts",
    #                      inverse_dynamics=True)

    rollouts_ds = file_dir + "/data/neg4to4_500k_rollouts__01-05-2023 05:12:23.npz"

    # nn = DynamicsNN(ds=arm,
    #                 n_nodes=128,
    #                 save_dir=file_dir + "/models",
    #                 decay_steps=1e8)
    # nn.train(rollouts_ds, n_epochs=150, lr=0.001,
    #          save_model=True, name="arm_tanh_rollouts_128", patience=6, batch_size=512)

    with np.load(rollouts_ds) as ds:
        x = ds["x_test"][:10]
        t = ds["y_test"][:10]

    import tensorflow as tf

    m = tf.keras.models.load_model("models/arm_tanh_rollouts_128_neg4to4__01-05-2023 05:16:43")
    print(m(x))
    print(t)
