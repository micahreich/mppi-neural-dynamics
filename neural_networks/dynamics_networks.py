import tensorflow as tf
import numpy as np
from datetime import datetime
import os


class TransferNN:
    def __init__(self, base_model, lr):
        self.base_model = base_model
        self.transfer_model = self.build_and_compile_model(lr)

    def build_and_compile_model(self, lr):
        base_model = self.base_model
        base_model.trainable = False

        out = tf.keras.layers.Dense(16, activation='tanh', name="transfer_layer")(base_model.layers[-2].output)
        out = tf.keras.layers.Dense(base_model.layers[-1].units, name="out_layer")(out)

        transfer_learning_model = tf.keras.models.Model(
            inputs=base_model.input, outputs=out, name="transfer_learning_dynamics_nn"
        )

        transfer_learning_model.compile(
            loss=tf.keras.losses.mean_absolute_error,
            optimizer=tf.keras.optimizers.Adam(lr)
        )
        transfer_learning_model.summary()

        return transfer_learning_model

    def train(self, x_data, y_data, n_epochs=1, verbose=0):
        history = self.transfer_model.fit(
            x_data,
            y_data,
            verbose=verbose, epochs=n_epochs,
            batch_size=len(x_data),
        )

        return history


class DynamicsNN:
    def __init__(self, ds, n_nodes, save_dir, decay_steps=None):
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
        self.decay_steps = decay_steps

    def load_dataset(self, npz_data_path):
        with np.load(npz_data_path) as data:
            self.train_examples = data['x_train']
            self.train_labels = data['y_train']
            self.test_examples = data['x_test']
            self.test_labels = data['y_test']

    def build_and_compile_model(self, norm, lr, cosine_lr=True):
        model = tf.keras.Sequential([
            norm,
            tf.keras.layers.Dense(self.n_nodes, activation='tanh'),
            tf.keras.layers.Dense(self.n_nodes, activation='tanh'),
            tf.keras.layers.Dense(self.system.nx // 2)
        ], name="dynamics_nn")

        if cosine_lr:
            assert self.decay_steps > 0

            lr = tf.keras.optimizers.schedules.CosineDecay(lr, self.decay_steps)

        model.compile(loss=tf.keras.losses.mean_absolute_error,
                      optimizer=tf.keras.optimizers.Adam(lr),
                      metrics=[tf.keras.losses.mean_absolute_error])

        model.summary()

        return model

    def train(self, npz_data_path, n_epochs=100, lr=1e-3,
              save_model=True, name=None, patience=3, cosine_lr=True):
        self.load_dataset(npz_data_path)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_examples))

        dynamics_model = self.build_and_compile_model(normalizer, lr=lr, cosine_lr=cosine_lr)

        print("[DynamicsNN] [Info] Beginning training with {} epochs".format(n_epochs))

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)]

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
