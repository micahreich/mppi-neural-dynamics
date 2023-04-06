import numpy as np
from typing import Callable
from enum import Enum


class ControlNoiseInit(Enum):
    LAST = 0
    RANDOM = 1
    ZERO = 2


class MPPIController:
    def __init__(self, n_rollouts: int, horizon_length: int, exploration_cov: np.ndarray, exploration_lambda: float,
                 nx: int, nu: int,
                 terminal_cost, state_cost, control_cost,
                 control_cov: np.ndarray,
                 evolve_state,
                 dt: float,
                 control_noise_initialization: ControlNoiseInit = ControlNoiseInit.LAST,
                 n_realized_controls=1,
                 control_range=None):
        """
         Model Predictive Path Integral Controller based on [1], [2]

        [1] https://openreview.net/pdf?id=ceOmpjMhlyS
        [2] https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf

        :param n_rollouts: Number of independent rollouts to simulate at each time step
        :param horizon_length: Number of time steps to simulate into the future for each rollout
        :param exploration_cov: 2D covariance matrix of shape (nu x nu) indicating the
        variance of the random control noise applied at each time step. Adjusting this value determines how aggressively
        the controller explores the state space when doing MPPI rollouts
        :param exploration_lambda: Positive scalar acting as a penalty on exploration of control space -- larger values
        allow more exploration
        :param nx: Number of elements in the state vector
        :param nu: Number of elements in the control vector
        :param terminal_cost: Function: (nx,) -> float; terminal state cost calculated at the end of
        each rollout
        :param state_cost: Function: (nx,) -> float; cost-to-go calculated once per horizon step per rollout
        :param control_cost: Function: (nu,) -> float; control cost calculated once per horizon
        step per rollout
        :param control_cov: 2D covariance matrix of shape (nu x nu) indicating the covariance of the actual applied
        control V where V ~ Normal(U, control_stddev)
        :param evolve_state: Function: ((nx nu), nx, nu) -> (nx,) takes in a (horizontal) stacked vector of the
        state and control along with the state and control dimensions and produces the next state
        :param dt: Time step of the controller
        :param control_noise_initialization: Method with which to initialize new control sequences
        :param n_realized_controls: Number of elements of the control sequence to be returned and executed on the
        system before resampling the control sequence
        :param control_range: Dict[str, (nu,)] with keys "min", "max", and "mean" indicating the min, max, and mean
        of control values to use in prediction
        """

        self._nx = nx
        self._nu = nu
        self._dt = dt

        self._n_rollouts = n_rollouts
        self._horizon_length = horizon_length
        self._exploration_cov = exploration_cov
        self._exploration_lambda = exploration_lambda

        self._default_control_seq = np.ones((self._horizon_length, self._nu))
        self._last_control_seq = self._default_control_seq

        self._evolve_state = evolve_state

        self._terminal_cost = terminal_cost
        self._state_cost = state_cost
        self._control_cost = control_cost

        self._control_cov = control_cov
        self._control_cov_inv = np.linalg.inv(control_cov)

        self._control_noise_initialization = control_noise_initialization
        self._n_realized_controls = n_realized_controls
        self._control_range = control_range

        if self._control_range is None:
            print("[MPPI] [Warn] No control range input. Assuming [-inf, inf] on all dimensions.")

    # def weight_rollouts(self, rollout_costs):
    #     """
    #     This function computes the weights of the rollouts based on the rollout costs using an exponential weighting scheme.
    #
    #     :param rollout_costs: A 1D numpy array representing the costs of the rollouts.
    #     :return: A 1D numpy array representing the normalized weights of the rollouts.
    #     """
    #
    #     min_cost = np.min(rollout_costs)
    #
    #     unnormed_rollout_costs = np.exp(-(1 / self._exploration_lambda) * (rollout_costs - min_cost))
    #     rollout_costs_sum = np.sum(unnormed_rollout_costs)
    #
    #     normalization_factor = 1 / rollout_costs_sum
    #
    #     return normalization_factor * unnormed_rollout_costs
    #
    # def resample_control_seq(self, rollout_costs, control_seq, control_noise):
    #     """
    #     Resamples the control sequence based on the weights of the rollout costs and the control noise.
    #
    #     :param rollout_costs: A 1D numpy array representing the costs of the rollouts.
    #     :param control_seq: A 3D numpy array (1, horizon_length, nu) representing the control sequence.
    #     :param control_noise: A 3D numpy array (n_rollouts, horizon_length, nu) representing the control noise.
    #
    #     :return: A 3D numpy array (1, horizon_length, nu) representing the resampled control sequence based on
    #     the previous sequence and rollout-weighted control noise.
    #     """
    #     # print("og seq", control_seq, control_seq.shape)
    #
    #     rollout_weights = self.weight_rollouts(rollout_costs)
    #     # print("rollout weights", rollout_weights, rollout_weights.shape)
    #
    #     repeated_weights = np.tile(rollout_weights.reshape((-1, 1, 1)),
    #                                reps=(1, self._horizon_length, self._nu))
    #
    #     # print("rollout weights (repd)", repeated_weights, repeated_weights.shape)
    #     # print("mult noise", control_noise * repeated_weights)
    #     weighted_noise = np.sum(control_noise * repeated_weights, axis=0).reshape((self._horizon_length, self._nu))
    #
    #     # print("reg noise", control_noise, control_noise.shape)
    #     # print("weighted_noise", weighted_noise, weighted_noise.shape)
    #
    #     resampled_control_seq = control_seq + weighted_noise
    #
    #     # print("resampled seq", resampled_control_seq, resampled_control_seq.shape)
    #
    #     return resampled_control_seq

    def roll_control_seq(self, control_seq):
        """
        Shifts the control sequence to the left by n_realized_controls items and
        initializes the remaining n_realized_controls at the end based on the input initialization method.

        :param control_seq: A 2D numpy array with shape (horizon_length, nu) representing the control sequence
        :return: A 2D numpy array representing the rolled control sequence with additional
        inputs generated based on the specified initialization method.
        """
        rolled_seq = np.roll(control_seq, -self._n_realized_controls, axis=0)
        size = (self._n_realized_controls, self._nu)

        if self._control_noise_initialization == ControlNoiseInit.RANDOM:
            if self._control_range is None:
                raise RuntimeError("For RANDOM control noise initialization, "
                                   "control_range must be supplied to the MPPI controller")

            new_controls = np.random.uniform(self._control_range["min"], self._control_range["max"], size=size)
        elif self._control_noise_initialization == ControlNoiseInit.LAST:
            new_controls = np.tile(control_seq[-self._n_realized_controls], reps=(self._n_realized_controls, 1))
        else:
            new_controls = np.zeros(size)

        rolled_seq[-self._n_realized_controls:] = new_controls

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

        assert (self._last_control_seq.shape == (self._horizon_length, self._nu))

        rollout_costs = np.zeros(self._n_rollouts)

        control_noise_seqs = np.zeros((self._horizon_length, self._nu, self._n_rollouts))

        for k in range(0, self._n_rollouts):
            curr_state = state

            control_noise_seq = np.random.multivariate_normal(mean=np.zeros(self._nu),
                                                              cov=self._exploration_cov,
                                                              size=self._horizon_length)
            control_noise_seqs[:, :, k] = control_noise_seq

            for t in range(0, self._horizon_length):
                curr_state = self._evolve_state(curr_state, self._last_control_seq[t] + control_noise_seq[t],
                                                dt=self._dt)
                rollout_costs[k] += self._state_cost(curr_state) + self._control_cost(self._last_control_seq[t],
                                                                                      control_noise_seq[t])
            rollout_costs[k] += self._terminal_cost(curr_state)

        beta = min(rollout_costs)
        scores = np.exp(-1/self._exploration_lambda * (rollout_costs - beta))

        scores_sum = np.sum(scores)
        weights = (1 / scores_sum) * scores

        for t in range(0, self._horizon_length):
            added_noise = 0

            for k in range(0, self._n_rollouts):
                added_noise += weights[k] * control_noise_seqs[t, :, k]
            self._last_control_seq[t] += added_noise

        u0 = self._last_control_seq[0]

        self._last_control_seq = np.roll(self._last_control_seq, -1, axis=0)
        self._last_control_seq[-1] = self._last_control_seq[-2]

        return u0
