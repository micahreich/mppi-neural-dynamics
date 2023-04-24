import tensorflow as tf
import numpy as np
from datetime import datetime
import os


class DynamicsNN:
    def __init__(self, ds, n_nodes, save_dir, learning_rate):
        self.system = ds

        try:
            assert (self.system.nx % 2 == 0)
        except AssertionError:
            print("[DynamicsNN] [Error] System's state space must be partitioned"
                  "into [x, x_dot] where |x| = |x_dot|")
            exit()

        self.n_nodes = n_nodes
        self.train_examples, self.train_labels, self.test_examples, self.test_labels = [None] * 4

        self.save_dir = save_dir
        self.learning_rate = learning_rate

    def load_dataset(self, npz_data_path):
        with np.load(npz_data_path) as data:
            self.train_examples = data['x_train']
            self.train_labels = data['y_train']
            self.test_examples = data['x_test']
            self.test_labels = data['y_test']

    def build_and_compile_model(self, norm):
        model = tf.keras.Sequential([
            norm,
            tf.keras.layers.Dense(self.n_nodes, activation='relu'),
            tf.keras.layers.Dense(self.n_nodes, activation='relu'),
            tf.keras.layers.Dense(self.system.nx // 2)
        ], name="dynamics_nn")

        model.compile(loss=tf.keras.losses.mean_absolute_error,
                      optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      metrics=[tf.keras.losses.mean_absolute_error])

        model.summary()

        return model

    def train(self, npz_data_path, n_epochs=100, save_model=True, name=None):
        self.load_dataset(npz_data_path)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_examples))

        dynamics_model = self.build_and_compile_model(normalizer)

        print("[DynamicsNN] [Info] Beginning training with {} epochs".format(n_epochs))

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)]

        history = dynamics_model.fit(
            self.train_examples,
            self.train_labels,
            validation_split=0.2,
            verbose=1, epochs=n_epochs,
            batch_size=1024,
            callbacks=callbacks)

        dynamics_model.evaluate(self.test_examples, self.test_labels, verbose=1)

        if save_model:
            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)

            dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            dir_fname_string = "{}/{}_model_{}nodes_{}".format(self.save_dir, str(self.system), self.n_nodes, dt_string)
            if name:
                dir_fname_string = "{}/{}__{}".format(self.save_dir, name, dt_string)

            dynamics_model.save(dir_fname_string)

        return history


class OnlineLearningDynamicsNN:
    def __init__(self, dnn, online_lr):
        self.dnn = dnn
        tf.keras.backend.set_value(self.dnn.optimizer.learning_rate, online_lr)

    def train_online(self, x, y):
        self.dnn.fit(
            x, y, verbose=0,
            epochs=1, batch_size=len(x)
        )
