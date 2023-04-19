import tensorflow as tf
import numpy as np
from datetime import datetime
import os


class DynamicsNN:
    def __init__(self, ds, n_nodes, npz_data_path, save_dir, learning_rate):
        self.system = ds
        self.n_nodes = n_nodes

        with np.load(npz_data_path) as data:
            self.train_examples = data['x_train']
            self.train_labels = data['y_train']
            self.test_examples = data['x_test']
            self.test_labels = data['y_test']

        self.save_dir = save_dir
        self.learning_rate = learning_rate

    def build_and_compile_model(self, norm):
        model = tf.keras.Sequential([
            norm,
            tf.keras.layers.Dense(self.n_nodes, activation='relu'),
            tf.keras.layers.Dense(self.n_nodes, activation='relu'),
            tf.keras.layers.Dense(self.system.nx)
        ], name="dynamics_nn")

        model.compile(loss=tf.keras.losses.mean_absolute_error,
                      optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      metrics=[tf.keras.losses.mean_absolute_error])

        model.summary()

        return model

    def train(self, n_epochs=100, save_model=True):
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_examples))

        dynamics_model = self.build_and_compile_model(normalizer)

        print("[DynamicsNN] [Info] Beginning training with {} epochs".format(n_epochs))

        def schedule(epoch_idx, lr):
            pass

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)]

        history = dynamics_model.fit(
            self.train_examples,
            self.train_labels,
            validation_split=0.2,
            verbose=1, epochs=n_epochs,
            callbacks=callbacks)

        dynamics_model.evaluate(self.test_examples, self.test_labels, verbose=1)

        if save_model:
            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)

            dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            dir_fname_string = "{}/{}_model_{}nodes_{}".format(self.save_dir, str(self.system), self.n_nodes, dt_string)

            dynamics_model.save(dir_fname_string)

        return history
