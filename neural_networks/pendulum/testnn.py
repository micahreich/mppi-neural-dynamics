import tensorflow as tf
from neural_networks.dataset_generator import DatasetGenerator
from neural_networks.dynamics_networks import TransferNN
import time

if __name__ == "__main__":
    # inverted_pendulum_ds = DatasetGenerator(ds=inverted_pendulum, save_dir=None)
    #
    # model = tf.keras.models.load_model("models/inverted_pendulum_model_32nodes_15-04-2023 02:42:35")
    # random_inputs, _ = inverted_pendulum_ds.sample_state_action_pair(n_samples=200)
    #
    # start = time.perf_counter()
    # res = model(random_inputs).numpy()
    # end = time.perf_counter()
    #
    # print("Predicted in {:.4}ms".format((end - start) * 1e3))

    pend_base_model = tf.keras.models.load_model("models/pendulum_tanh_32_rollouts__26-04-2023 02:34:02")
    pend_transfer_model = TransferNN(base_model=pend_base_model).build_and_compile_model(1e-3)
