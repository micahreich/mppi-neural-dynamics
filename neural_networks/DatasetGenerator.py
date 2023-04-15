import os

import numpy as np
import scipy
from tempfile import TemporaryFile
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf


def npz_to_tf_dataset(npz_data_path):
    with np.load(npz_data_path) as data:
        train_examples = data['x_train']
        train_labels = data['y_train']
        test_examples = data['x_test']
        test_labels = data['y_test']

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    return train_dataset, test_dataset


class DatasetGenerator:
    def __init__(self, ds, save_dir):
        self.system = ds
        self.save_dir = save_dir
        self.latest_dataset_fname = None

    def sample_actions(self, n_samples, u_lo, u_hi):
        actions = np.empty(shape=(n_samples, self.system.nu))

        for i in range(self.system.nu):
            actions[:, i] = np.random.uniform(low=u_lo[i], high=u_hi[i], size=n_samples)
        return actions

    def sample_initial_states(self, n_samples, x_lo, x_hi):
        states = np.empty(shape=(n_samples, self.system.nx))

        for i in range(self.system.nx):
            states[:, i] = np.random.uniform(low=x_lo[i], high=x_hi[i], size=n_samples)
        return states

    def sample_state_action_pair(self, n_samples):
        data_x = np.empty(shape=(n_samples, self.system.nx + self.system.nu))
        data_y = np.empty(shape=(n_samples, self.system.nx))

        actions = self.sample_actions(n_samples, self.system.u_lo, self.system.u_hi)
        initial_states = self.sample_initial_states(n_samples, self.system.x_lo, self.system.x_hi)

        for i in tqdm(range(n_samples)):
            initial_state, action = initial_states[i], actions[i]

            dynamics_ode = self.system.dynamics(action)
            next_state = scipy.integrate.odeint(dynamics_ode, initial_state, np.array([0, self.system.dt]))[1]

            data_x[i] = np.hstack((initial_state, action))
            data_y[i] = next_state

        return data_x, data_y

    def generate_dataset(self, n_samples, train_percentage, name=None):
        print("[DatasetGenerator] [Info] Generating {} samples with: {:.1f}% train {:.1f}% test".format(
            n_samples,
            train_percentage * 100,
            (1 - train_percentage) * 100))

        data_x, data_y = self.sample_state_action_pair(n_samples)

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        try:
            n_train_samples = int(train_percentage * len(data_x))

            x_train, y_train = data_x[:n_train_samples], data_y[:n_train_samples]
            x_test, y_test = data_x[n_train_samples:], data_y[n_train_samples:]

            dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            dir_fname_string = "{}/{}_data_{}".format(self.save_dir, str(self.system), dt_string)

            if name:
                dir_fname_string = "{}/{}_{}".format(self.save_dir, name, dt_string)

            np.savez(dir_fname_string, x_train=x_train, y_train=y_train,
                                       x_test=x_test, y_test=y_test)

            file_size = os.path.getsize("{}.npz".format(dir_fname_string))
            print("[DatasetGenerator] [Info] Saved {} dataset with size {} MB".format(str(self.system),
                                                                                      int(file_size / 1e6)))

            dataset_fname = "{}.npz".format(dir_fname_string)
            self.latest_dataset_fname = dataset_fname

            return dataset_fname
        except FileNotFoundError:
            print("[DatasetGenerator] [Error] Error saving dataset to disk")

    def inspect_datasets(self, dataset_fname=None):
        if not dataset_fname:
            return np.load(self.latest_dataset_fname)
        return np.load(dataset_fname)
