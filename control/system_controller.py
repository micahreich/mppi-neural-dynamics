import sys
sys.path.append("..")

from MPPIController import MPPIController, ControlNoiseInit


class SystemController:
    def __init__(self, ds,
                 n_rollouts, horizon_length, exploration_cov, exploration_lambda,
                 state_cost, terminal_cost, control_cost=None,
                 nn_model=None, control_range=None, control_noise_initialization=ControlNoiseInit.RANDOM):
        evolve_state = ds.integrator.step
        nn_dynamics = False

        if nn_model is not None:
            nn_dynamics = True

            def nn_evolve_state(nn_input):
                states, controls = nn_input[:, :ds.nx], nn_input[:, ds.nx:]
                x_ddots = nn_model(nn_input)

                x_dots = states[:, 1::2]

                next_states = states
                next_states[:, ::2] += x_dots * ds.dt + 1 / 2 * x_ddots * (ds.dt ** 2)
                next_states[:, 1::2] += x_ddots * ds.dt

                return next_states

            evolve_state = nn_evolve_state

        self.controller = MPPIController(
            n_rollouts=n_rollouts,
            horizon_length=horizon_length,
            exploration_cov=exploration_cov,
            exploration_lambda=exploration_lambda,
            nx=ds.nx,
            nu=ds.nu,
            terminal_cost=terminal_cost,
            state_cost=state_cost,
            evolve_state=evolve_state,
            control_cost=control_cost,
            dt=ds.dt,
            control_range=control_range,
            control_noise_initialization=control_noise_initialization,
            nn_dynamics=nn_dynamics
        )

    def build(self): return self.controller
