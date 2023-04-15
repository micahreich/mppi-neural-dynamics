from neural_networks.DatasetGenerator import DatasetGenerator
from neural_networks.DynamicsNN import DynamicsNN
from InvertedPendulum import InvertedPendulum
import os

inverted_pendulum = InvertedPendulum(m=1, l=1, b=0.1, g=-9.81, dt=1 / 10)
file_dir = os.path.dirname(os.path.realpath(__file__))


def generate_dataset(n_samples, train_percentage):
    SAVE_DIR = file_dir + "/data"

    # Create inverted pendulum class and dataset generator
    inverted_pendulum_ds = DatasetGenerator(ds=inverted_pendulum, save_dir=SAVE_DIR)

    # Generate data points and save to .npz file
    dataset_fname = inverted_pendulum_ds.generate_dataset(n_samples=n_samples, train_percentage=train_percentage)

    # Inspect saved data
    with inverted_pendulum_ds.inspect_datasets() as ds:
        print(len(ds["x_train"]), len(ds["x_test"]))

    return dataset_fname


if __name__ == "__main__":
    # dataset_fname = generate_dataset(n_samples=200000, train_percentage=0.7)
    dataset_fname = file_dir + "/data/inverted_pendulum_data_15-04-2023 02:35:20.npz"

    nn = DynamicsNN(ds=inverted_pendulum,
                    n_nodes=32,
                    npz_data_path=dataset_fname,
                    save_dir=file_dir + "/models")

    nn.train(n_epochs=6)
