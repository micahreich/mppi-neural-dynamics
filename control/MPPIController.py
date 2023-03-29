import numpy as np
from typing import Callable


class MPPIController:
    def __init__(self, n_rollouts: int, horizon_length: int, exploration_covariance: np.ndarray,
                 state_shape: int, control_shape: int,
                 terminal_cost, state_cost, control_cost,
                 control_stddev: np.ndarray, control_lambda: float,
                 evolve_state: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 dt: float,
                 control_noise_initialization="LAST",
                 n_realized_controls=1,
                 control_range=None):
        """
         Model Predictive Path Integral Controller based on [1], [2]

        [1] https://openreview.net/pdf?id=ceOmpjMhlyS
        [2] https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf

        :param n_rollouts: Number of independent rollouts to simulate at each time step
        :param horizon_length: Number of time steps to simulate into the future for each rollout
        :param exploration_covariance: 2D covariance matrix of shape (control_shape x control_shape) indicating the
        variance of the random control noise applied at each time step. Adjusting this value determines how aggressively
        the controller explores the state space when doing MPPI rollouts
        :param state_shape: Number of elements in the state vector
        :param control_shape: Number of elements in the control vector
        :param terminal_cost: Function f: (state_shape,) -> float; terminal state cost calculated at the end of
        each rollout
        :param state_cost: Function g: (state_shape,) -> float; cost-to-go calculated once per horizon step per rollout
        :param control_cost: Function h: (control_shape,) -> float; control cost calculated once per horizon
        step per rollout
        :param control_stddev: ?
        :param control_lambda: ?
        :param evolve_state: 
        :param dt:
        :param control_noise_initialization:
        :param n_realized_controls:
        :param control_range:
        """

        self._state_shape = state_shape
        self._control_shape = control_shape

        self._n_rollouts = n_rollouts
        self._horizon_length = horizon_length
        self._exploration_covariance = exploration_covariance

        self._default_control_seq = np.zeros(self._horizon_length)
        self._last_control_seq = self._default_control_seq

        self._evolve_state = evolve_state

        self._terminal_cost = terminal_cost
        self._state_cost = state_cost
        self._control_cost = control_cost

        self._control_stddev = control_stddev
        self._control_lambda = control_lambda

        self._control_noise_initialization = control_noise_initialization
        self._n_realized_controls = n_realized_controls
        self._control_range = control_range

    def weight_rollout_costs(self, rollout_costs):
        """
        This function computes the weights of the rollouts based on the rollout costs using an exponential weighting scheme.

        :param rollout_costs: A 1D numpy array representing the costs of the rollouts.
        :return: A 1D numpy array representing the normalized weights of the rollouts.
        """

        min_cost = np.min(rollout_costs)

        unnormed_rollout_costs = np.exp(-(1 / self._control_lambda) * (rollout_costs - min_cost))
        normalization_factor = 1 / np.sum(unnormed_rollout_costs)

        return normalization_factor * unnormed_rollout_costs

    def resample_control_seq(self, rollout_costs, control_seq, control_noise):
        """
        Resamples the control sequence based on the weights of the rollout costs and the control noise.

        :param rollout_costs: A 1D numpy array representing the costs of the rollouts.
        :param control_seq: A 1D numpy array representing the control sequence.
        :param control_noise: A 2D numpy array representing the control noise.
        :return: A 1D numpy array representing the resampled control sequence based on the rollout weights and control noise.
        """

        rollout_weights = self.weight_rollout_costs(rollout_costs)

        repeated_weights = np.repeat(rollout_weights.reshape((-1, 1)),
                                     repeats=self._horizon_length, axis=1)

        weighted_noise = np.sum(control_noise * repeated_weights, axis=0)
        resampled_control_seq = control_seq + weighted_noise

        return resampled_control_seq

    def roll_control_seq(self, control_seq):
        """
        Shifts the control sequence to the left by n_realized_controls items and
        initializes the remaining n_realized_controls at the end based on the input initialization method.

        :param control_seq: A 1D numpy array representing the control sequence
        :return: A 1D numpy array representing the rolled control sequence with additional
        inputs generated based on the specified initialization method.
        """

        rolled_seq = np.roll(control_seq, -self._n_realized_controls)

        for i in range(self._horizon_length - self._n_realized_controls, self._horizon_length):
            if self._control_noise_initialization == "RANDOM":
                if self._control_range is None:
                    raise RuntimeError("For RANDOM control noise initialization, "
                                       "control_range must be supplied to the MPPI controller")

                control_lo, control_hi = self._control_range
                rolled_seq[i] = np.random.uniform(control_lo, control_hi, self._control_shape)
            elif self._control_noise_initialization == "LAST":
                rolled_seq[i] = rolled_seq[control_seq[-1]]
            else:
                rolled_seq[i] = np.zeros(self._control_shape)

        return rolled_seq

    def step(self, state):
        """
        This function performs a single step of the MPPI controller.
        It generates random control noise, evolves the state using the last control sequence and the random
        control noise, and calculates the costs of the rollouts. It then resamples the control sequence based on the
        weights of the rollouts and rolls the control sequence. Finally, it returns the first n_realized_controls
        elements of the resampled control sequence.

        :param state: A 1D numpy array representing the current state.
        :return: The first n_realized_controls elements of the resampled control sequence.

        Represent random control noise as a np array with shape (K, T, U) where U is the
        shape of the control input.
        """

        control_noise = np.random.multivariate_normal(mean=np.zeros(self._control_shape),
                                                      cov=self._exploration_covariance,
                                                      size=(self._n_rollouts, self._horizon_length))

        rollout_costs = np.zeros(self._n_rollouts)
        last_state = state

        last_states = np.tile(state, reps=(self._n_rollouts, 1))
        control_seqs = np.tile(self._last_control_seq.reshape((1, self._horizon_length, self._control_shape)),
                               reps=(self._n_rollouts, 1, 1))

        for t in range(0, self._horizon_length):
            current_control = control_seqs[:, t]
            current_control_noise = control_noise[:, t]
            noisy_controls = current_control + current_control_noise

            # Evolve the states of each rollout for the current time step in parallel
            evolve_state_args = np.hstack((last_state, noisy_controls))
            current_states = np.apply_along_axis(self._evolve_state, axis=1, arr=evolve_state_args,
                                                 state_dim=self._state_shape, control_dim=self._control_shape)

            # Compute the state costs of each rollout for the current time step in parallel
            rollout_costs += np.apply_along_axis(self._state_cost, axis=1, arr=current_states)

            # Compute the control costs of each rollout for the current time step in parallel
            controls_cost_args = np.hstack((current_control, current_control_noise))
            rollout_costs += np.apply_along_axis(self._control_cost, axis=1, arr=controls_cost_args,
                                                 sigma=self._control_stddev, c=self._control_lambda)

            last_states = current_states

        rollout_costs += np.apply_along_axis(self._terminal_cost, axis=1, arr=last_states)

        resampled_control_seq = self.resample_control_seq(rollout_costs, self._last_control_seq, control_noise)
        self._last_control_seq = self.roll_control_seq(resampled_control_seq)

        return resampled_control_seq[:self._n_realized_controls]
