from neural_networks.dataset_generator import DatasetGenerator
from neural_networks.dynamics_networks import DynamicsNN
from systems.dynamical_systems import Pendulum
import os
import numpy as np

inverted_pendulum = Pendulum(m=1, l=1, b=0.1, dt=1 / 10)
file_dir = os.path.dirname(os.path.realpath(__file__))


def generate_dataset(n_samples, sample_method, horizon_length, train_percentage, name):
    SAVE_DIR = file_dir + "/data"

    # Create inverted pendulum class and dataset generator
    pendulum_ds = DatasetGenerator(ds=inverted_pendulum, save_dir=SAVE_DIR,
                                            sample_method=sample_method)

    # Generate data points and save to .npz file
    dataset_fname = pendulum_ds.generate_dataset(n_samples=n_samples, horizon_length=horizon_length,
                                                          train_percentage=train_percentage, name=name)

    # Inspect saved data
    with pendulum_ds.inspect_datasets() as ds:
        print("Training dataset size: {}     Testing dataset size: {}".format(len(ds["x_train"]), len(ds["x_test"])))

    return dataset_fname


if __name__ == "__main__":
    # _ = generate_dataset(n_samples=500_000, sample_method="random rollouts",
    #                      horizon_length=10, train_percentage=0.8, name="500k_rollouts_2to2")

    rollouts_ds = file_dir + "/data/500k_rollouts_2to2__26-04-2023 02:32:28.npz"
    nn = DynamicsNN(ds=inverted_pendulum,
                    n_nodes=16,
                    save_dir=file_dir + "/comparison_models", decay_steps=4e7)

    nn.train(rollouts_ds, n_epochs=150, lr=0.001,
             save_model=True, name="pendulum_tanh_16_rollouts", patience=4)
