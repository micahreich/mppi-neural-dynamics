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
                 terminal_cost, state_cost,
                 evolve_state,
                 dt: float,
                 control_noise_initialization: ControlNoiseInit = ControlNoiseInit.LAST,
                 n_realized_controls=1,
                 control_range=None,
                 control_cost=None):
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
        :param terminal_cost: function: (ndarray(nx)) -> float32
        :param state_cost: function: (ndarray(nx)) -> float32
        :param control_cost: function: (ndarray(nu), ndarray(nu)) -> float32
        :param evolve_state: function: (ndarray(nx), ndarray(nu)) -> ndarray(nx)
        :param control_cov: 2D covariance matrix of shape (nu x nu) indicating the covariance of the actual applied
        control V where V ~ Normal(U, control_stddev)
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

        assert (exploration_cov.shape == (self._nu, self._nu))
        self._exploration_cov = exploration_cov
        self._exploration_lambda = exploration_lambda

        self._default_control_seq = np.zeros((self._horizon_length, self._nu))
        self._last_control_seq = self._default_control_seq

        self._evolve_state = np.vectorize(evolve_state, signature="(nx),(nu),()->(nx)")
        self._terminal_cost = np.vectorize(terminal_cost, signature="(nx)->()")
        self._state_cost = np.vectorize(state_cost, signature="(nx)->()")

        self._exploration_cov_inv = np.linalg.inv(self._exploration_cov)

        if not control_cost:
            def default_control_cost(u, noise):
                return self._exploration_lambda * np.dot(
                    u,
                    np.dot(self._exploration_cov_inv, noise)
                )

            control_cost = default_control_cost

        self._control_cost = np.vectorize(control_cost, signature="(nu),(nu)->()")

        self._control_noise_initialization = control_noise_initialization
        self._n_realized_controls = n_realized_controls
        self._control_range = control_range

        if self._control_range is None:
            print("[MPPI] [Warn] No control range input. Assuming [-inf, inf] on all dimensions.")
        else:
            assert (self._control_range["min"].shape == self._control_range["min"].shape == (self._nu,))

            self._rollout_control_range = {
                "min": np.tile(self._control_range["min"].reshape(1, -1), (self._n_rollouts, 1)),
                "max": np.tile(self._control_range["max"].reshape(1, -1), (self._n_rollouts, 1))
            }

    def score_rollouts(self, rollout_cumcosts):
        """
        Score each rollout according to its cumulative cost

        :param rollout_cumcosts: Array of cumulative rollout costs
        :return: Array of scores, parallel with the original cost array
        """

        assert (rollout_cumcosts.shape == (self._n_rollouts,))

        beta = min(rollout_cumcosts)
        scores = np.exp(-1 / self._exploration_lambda * (rollout_cumcosts - beta))

        scores_sum = np.sum(scores)
        return (1 / scores_sum) * scores

    def weight_rollout_noise(self, rollout_noise_u, weights):
        """
        Add weighted control noise to nominal control sequence to get new, optimized control sequence

        :param rollout_noise_u: Gaussian noise used in the MPC rollout phase
        :param weights: Weights for each rollout
        :return: Modified control sequence incorporating the control noise, favoring high-scoring  perturbations
        """

        assert (rollout_noise_u.shape == (self._n_rollouts, self._nu, self._horizon_length) and
                weights.shape == (self._n_rollouts,))

        weights = np.tile(weights.reshape((-1, 1, 1)), (1, self._nu, self._horizon_length))
        weighted_noise_u = rollout_noise_u * weights
        weighted_noise_u = np.sum(weighted_noise_u, axis=0).T

        return self._last_control_seq + weighted_noise_u

    def shift_control_seq(self, control_seq):
        """
        Shift the nominal control sequence ahead by 1 index
        :param control_seq: Control sequence to be shifted
        :return: Shifted control sequence with the last item initialized as defined by the initialization method
        """

        control_seq = np.roll(control_seq, -1, axis=0)

        if self._control_noise_initialization == ControlNoiseInit.RANDOM:
            if self._control_range is None:
                raise RuntimeError("For RANDOM control noise initialization, "
                                   "control_range must be supplied to the MPPI controller")

            new_controls = np.random.uniform(self._control_range["min"], self._control_range["max"], size=self._nu)
        elif self._control_noise_initialization == ControlNoiseInit.LAST:
            new_controls = control_seq[-2]
        else:
            new_controls = np.zeros(self._nu)

        control_seq[-1] = new_controls
        return control_seq

    def step(self, state):
        """
        Perform a single step of the MPPI controller.
        Step generates random control noise, evolves the state using the last control sequence and the random
        control noise, and calculates the costs of the rollouts. It then resamples the control sequence based on the
        weights of the rollouts and rolls the control sequence. Finally, it returns the first n_realized_controls
        elements of the resampled control sequence.

        :param state: A 1D numpy array representing the current state.
        :return: The first n_realized_controls elements of the resampled control sequence.

        Represent random control noise as a np array with shape (K, T, U) where U is the
        shape of the control input.
        """
        assert (state.shape == (self._nx,))
        assert (self._last_control_seq.shape == (self._horizon_length, self._nu))

        rollout_current_states = np.tile(state, (self._n_rollouts, 1))
        rollout_cumcosts = np.zeros(self._n_rollouts)

        rollouts_noise_u = np.random.multivariate_normal(mean=np.zeros(self._nu),
                                                         cov=self._exploration_cov,
                                                         size=(self._n_rollouts, self._horizon_length)).swapaxes(1, 2)

        for t in range(0, self._horizon_length):
            rollout_noise_u = rollouts_noise_u[:, :, t]
            rollout_nominal_u = np.tile(self._last_control_seq[t].reshape(1, -1),
                                        (self._n_rollouts, 1))

            rollout_current_u = rollout_nominal_u + rollout_noise_u

            if self._control_range:
                rollout_current_u = np.clip(rollout_current_u,
                                            self._rollout_control_range["min"], self._rollout_control_range["max"])

            # Evolve states in vectorized form
            rollout_current_states = self._evolve_state(rollout_current_states, rollout_current_u, self._dt)

            # Calculate current time-step state cost
            rollout_cumcosts += self._state_cost(rollout_current_states)

            # Calculate current time-step control cost
            rollout_cumcosts += self._control_cost(rollout_nominal_u, rollout_noise_u)

        # Calculate terminal cost
        rollout_cumcosts += self._terminal_cost(rollout_current_states)

        # Weight control noise and add to last nominal control sequence
        rollout_scores = self.score_rollouts(rollout_cumcosts)
        self._last_control_seq = self.weight_rollout_noise(rollouts_noise_u, weights=rollout_scores)

        # Roll forward the nominal controls and return the first action
        u0 = self._last_control_seq[0]

        if self._control_range:
            u0 = np.clip(u0, self._control_range["min"], self._control_range["max"])

        self._last_control_seq = self.shift_control_seq(self._last_control_seq)

        return u0