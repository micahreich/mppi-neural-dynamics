import tensorflow as tf
from neural_networks.dataset_generator import DatasetGenerator
from InvertedPendulum import InvertedPendulum
import time

inverted_pendulum = InvertedPendulum(m=1, l=1, b=0.1, g=-9.81, dt=1 / 10)

if __name__ == "__main__":
    inverted_pendulum_ds = DatasetGenerator(ds=inverted_pendulum, save_dir=None)

    model = tf.keras.models.load_model("models/inverted_pendulum_model_32nodes_15-04-2023 02:42:35")
    random_inputs, _ = inverted_pendulum_ds.sample_state_action_pair(n_samples=200)

    start = time.perf_counter()
    res = model(random_inputs).numpy()
    end = time.perf_counter()

    print("Predicted in {:.4}ms".format((end - start) * 1e3))
