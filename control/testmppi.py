import numpy as np


def _evolve_state(x, state_dim, control_dim):
    states, controls = x[:state_dim], x[state_dim:]
    return states+1

def _state_cost(x):
    return -1

def _control_cost(u):
    return -5

def _terminal_cost(x):
    return 0.1


_control_shape = 3
_state_shape = 6
_exploration_covariance = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

_n_rollouts = 5
_horizon_length = 10

_last_control_seq = np.arange(_horizon_length * _control_shape).reshape(_horizon_length, _control_shape)

state = np.ones(_state_shape)

control_noise = np.random.multivariate_normal(mean=np.zeros(_control_shape),
                                              cov=_exploration_covariance,
                                              size=(_n_rollouts, _horizon_length))
# print("Control noise:", control_noise)

last_states = np.tile(state, reps=(_n_rollouts, 1))
control_seqs = np.tile(_last_control_seq.reshape((1, _horizon_length, _control_shape)),
                       reps=(_n_rollouts, 1, 1))

rollout_costs = np.zeros(_n_rollouts)

for t in range(0, _horizon_length):
    current_control = control_seqs[:, t]
    current_control_noise = control_noise[:, t]
    noisy_controls = current_control + current_control_noise

    # Evolve the states of each rollout for the current time step in parallel
    evolve_state_args = np.hstack((last_states, noisy_controls))
    current_states = np.apply_along_axis(_evolve_state, axis=1, arr=evolve_state_args,
                                         state_dim=_state_shape, control_dim=_control_shape)

    # Compute the state costs of each rollout for the current time step in parallel
    rollout_costs += np.apply_along_axis(_state_cost, axis=1, arr=current_states)

    # Compute the control costs of each rollout for the current time step in parallel
    controls_cost_args = np.hstack((current_control, current_control_noise))
    rollout_costs += np.apply_along_axis(_control_cost, axis=1, arr=controls_cost_args)

    last_states = current_states

rollout_costs += np.apply_along_axis(_terminal_cost, axis=1, arr=last_states)
print(rollout_costs)