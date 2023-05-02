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
    def __init__(self, ds, save_dir, sample_method="random singles", inverse_dynamics=False):
        """
        Dataset generator for neural dynamics models. Data produced by this API is trained with a
        DNN to predict the acceleration of a dynamical system given the current state and control input

        :param ds: dynamical system class from '/systems' directory
        :param save_dir: directory to save the .npz dataset to
        :param sample_method: method to use for data set sampling. Let n be the # of points in the dataset.
        :param inverse_dynamics: indicate whether training an inverse dynamics model. If true, we must use
        random singles generation method

        'random singles' indicates sampling n random initial states and actions, integrating the system forward by
        dt, and using the resulting state to determine the state evolution.

        'random rollouts' indicates sampling k initial states (where k = number of rollouts) and n random controls,
        where each of the k initial states are rolled out (T = n // k) times, and each (x_t, x_{t+1}) pair is used
        to determine the state evolutions.
        """
        self.system = ds

        try:
            assert (self.system.nx % 2 == 0)
        except AssertionError:
            print("[DatasetGenerator] [Error] System's state space must be partitioned"
                  "into [x, x_dot] where |x| = |x_dot|")
            exit()

        self.save_dir = save_dir
        self.latest_dataset_fname = None
        self.sample_method = sample_method

        self.batched_ensure_state = np.vectorize(self.system.ensure_state, signature="(nx)->(nx)")
        self.batched_next_state = np.vectorize(self.system.integrator.step, signature="(nx),(nu)->(nx)")

        self.inverse_dynamics = inverse_dynamics
        if self.inverse_dynamics:
            self.batched_inverse_dynamics = np.vectorize(self.system.inverse_dynamics, signature="(nx),(nu)->(nu)")
        
    def _uniform_2d_bycol(self, nr, nc, lo, hi):
        x = np.empty(shape=(nr, nc))

        for i in range(nc):
            x[:, i] = np.random.uniform(low=lo[i], high=hi[i], size=nr)
        return x

    def sample_actions(self, n_samples, u_lo, u_hi):
        return self._uniform_2d_bycol(nr=n_samples, nc=self.system.nu, lo=u_lo, hi=u_hi)

    def sample_initial_states(self, n_samples, x_lo, x_hi):
        return self._uniform_2d_bycol(nr=n_samples, nc=self.system.nx, lo=x_lo, hi=x_hi)
    
    # def sample_accelerations(self, n_samples, a_lo, a_hi):
    #     return self._uniform_2d_bycol(nr=n_samples, nc=self.system.nu, lo=a_lo, hi=a_hi)

    def sample_pairs(self, n_samples, horizon_length=None):

        if self.sample_method == "random rollouts" and \
           n_samples/horizon_length != n_samples//horizon_length:
            rounded_n_samples = int(horizon_length * np.ceil(n_samples / horizon_length))

            print("[DatasetGenerator] [Error] n_samples of {} does not generate a whole number of rollouts with "
                  "rollout_length {}. "
                  "Rounding n_samples up to {}".format(n_samples, horizon_length, rounded_n_samples))
            n_samples = rounded_n_samples

        data_x = np.empty(shape=(n_samples, self.system.nx + self.system.nu))
        data_y = np.empty(shape=(n_samples, self.system.nx // 2))

        if self.sample_method == "random singles":
            initial_states = self.batched_ensure_state(
                self.sample_initial_states(n_samples, self.system.x_lo, self.system.x_hi)
            )
            
            if self.inverse_dynamics:
                actions = self.sample_actions(n_samples, self.system.accel_lo, self.system.accel_hi)
                torques = self.batched_inverse_dynamics(initial_states, actions)

                data_x = np.hstack((initial_states, actions))
                data_y = torques
            else:
                actions = self.sample_actions(n_samples, self.system.u_lo, self.system.u_hi)
                next_states = self.batched_next_state(initial_states, actions)

                accelerations = (next_states[:, 1::2] - initial_states[:, 1::2]) / self.system.dt

                data_x = np.hstack((initial_states, actions))
                data_y = accelerations
            
            # for i in tqdm(range(n_samples)):
            #     initial_state, action = tstep_states[i], actions[i]

            #     if self.inverse_dynamics:
            #         torque = self.system.inverse_dynamics(initial_state, action)
            #         next_state = self.system.integrator.step(initial_state, torque)

            #         data_x[i] = np.hstack((initial_state, action))
            #         data_y[i] = torque
            #     else:
            #         next_state = self.system.integrator.step(initial_state, action)

            #         acceleration = (next_state[1::2] - initial_state[1::2]) / self.system.dt

            #         data_x[i] = np.hstack((initial_state, action))
            #         data_y[i] = acceleration
        elif self.sample_method == "random rollouts":
            n_rollouts = n_samples // horizon_length
            print("[DatasetGenerator] [Info] Generating {} rollouts of length {}.".format(n_rollouts, horizon_length))

            tstep_states = self.batched_ensure_state(
                self.sample_initial_states(n_rollouts, self.system.x_lo, self.system.x_hi)
            )

            for t in tqdm(range(horizon_length)):
                if self.inverse_dynamics:
                    tstep_actions = self.sample_actions(n_rollouts, self.system.accel_lo, self.system.accel_hi)
                    tstep_torques = self.batched_inverse_dynamics(tstep_states, tstep_actions)

                    tstep_next_states = self.batched_ensure_state(
                        self.batched_next_state(tstep_states, tstep_torques)
                    )

                    data_x[t * n_rollouts:(t + 1) * n_rollouts, :] = np.hstack((tstep_states, tstep_actions))
                    data_y[t * n_rollouts:(t + 1) * n_rollouts, :] = tstep_torques
                else:
                    tstep_actions = self.sample_actions(n_rollouts, self.system.u_lo, self.system.u_hi)
                    tstep_next_states = self.batched_ensure_state(
                        self.batched_next_state(state=tstep_states, control=tstep_actions)
                    )

                    tstep_velocities = tstep_states[:, 1::2]
                    tstep_next_velocities = tstep_next_states[:, 1::2]

                    accelerations = (tstep_next_velocities - tstep_velocities) / self.system.dt

                    data_x[t * n_rollouts:(t + 1) * n_rollouts, :] = np.hstack((tstep_states, tstep_actions))
                    data_y[t * n_rollouts:(t + 1) * n_rollouts, :] = accelerations

                tstep_states = tstep_next_states
        else:
            print("[DatasetGenerator] [Error] Invalid sample type.")
            exit()

        return data_x, data_y

    def generate_dataset(self, n_samples, train_percentage, name=None, horizon_length=None):
        if self.sample_method == "random rollouts" and not horizon_length:
            print("[DatasetGenerator] [Error] For sample_method = 'random rollouts', horizon_length must be of type"
                  " int > 0")
            exit()

        print("[DatasetGenerator] [Info] Generating {} samples with: {:.1f}% train {:.1f}% test".format(
            n_samples,
            train_percentage * 100,
            (1 - train_percentage) * 100))

        data_x, data_y = self.sample_pairs(n_samples, horizon_length)

        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        try:
            n_train_samples = int(train_percentage * len(data_x))

            x_train, y_train = data_x[:n_train_samples], data_y[:n_train_samples]
            x_test, y_test = data_x[n_train_samples:], data_y[n_train_samples:]

            dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            dir_fname_string = "{}/{}-{}-samples__{}".format(self.save_dir, str(self.system), n_samples, dt_string)

            if name:
                dir_fname_string = "{}/{}__{}".format(self.save_dir, name, dt_string)

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
