import sys
sys.path.append("..")

from system_controller import SystemController, ControlNoiseInit
from utils import transform_angle_error, transform_02pi_to_negpospi, Simulator, ParametricPath, Trajectory
from neural_networks.dynamics_networks import TransferNN

import numpy as np
import time
import matplotlib.pyplot as plt
from time import perf_counter
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

plt.style.use(["science", "vibrant"])
import tensorflow as tf

class NNGTComparison:
    def __init__(self, ds, gt_controller, nn_controller):
        self.ds = ds
        self.gt_controller = gt_controller
        self.nn_controller = nn_controller
    
    def compare_trajectory_costs(self, n_samples, initial_state, simulation_length):
        state_cost = self.gt_controller.controller._state_cost
        control_cost = self.gt_controller.controller._control_cost

        nn_cumcosts = np.zeros(n_samples)
        gt_cumcosts = np.zeros(n_samples)

        nn_env = Simulator(self.ds, controller=self.nn_controller)
        gt_env = Simulator(self.ds, controller=self.gt_controller)

        for i in tqdm(range(n_samples)):
            nn_states, nn_controls, _ = nn_env.run(simulation_length=simulation_length, initial_state=initial_state,
                                                   controlled=True)
            nn_cumcosts[i] = np.sum(state_cost(nn_states) + control_cost(nn_controls))

            gt_states, gt_controls, _ = gt_env.run(simulation_length=simulation_length, initial_state=initial_state,
                                                   controlled=True)
            gt_cumcosts[i] = np.sum(state_cost(gt_states) + control_cost(gt_controls))
        
        dt_string = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        fname = "comparison_data/{}_comparison_{}".format(str(self.ds), dt_string)

        np.savez(fname, nn_cumcosts=nn_cumcosts, gt_cumcosts=gt_cumcosts)

        mean_nn_cost, std_nn_cost = np.mean(nn_cumcosts), np.std(nn_cumcosts)
        mean_gt_cost, std_gt_cost = np.mean(gt_cumcosts), np.std(gt_cumcosts)

        print("[Comparison] Comparison for system ({}) \n"
              "             GT cost mean: {:.4f}, GT cost variance: {:.4f} \n"
              "             NN cost mean: {:.4f}, NN cost variance: {:.4f} \n"
              "             E[C_NN] / E[C_GT] = {:.4f}".format(
                str(self.ds),
                mean_gt_cost, std_gt_cost,
                mean_nn_cost, std_nn_cost,
                mean_nn_cost / mean_gt_cost
            ))

