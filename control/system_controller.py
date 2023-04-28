import sys
sys.path.append("..")

from mppi_controller import MPPIController, ControlNoiseInit
import numpy as np


class SystemController:
    def __init__(self, ds,
                 n_rollouts, horizon_length, exploration_cov, exploration_lambda,
                 state_cost, terminal_cost, control_cost=None,
                 nn_model=None, control_range=None, control_noise_initialization=ControlNoiseInit.RANDOM,
                 include_null_controls=True, inverse_dyn_control=False):
        self.ds = ds
        self.inverse_dyn_control = inverse_dyn_control
        self.nn_model = nn_model

        # Default is forward dynamics with dynamics simulation inside MPC
        evolve_state = ds.integrator.step
        nn_dynamics = evolve_state_batched = False

        if nn_model is not None and not inverse_dyn_control:
            # Forward dynamics with dynamics simulation inside MPC using NN for dynamics model
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
            evolve_state_batched = True

        elif nn_model is None and inverse_dyn_control:
            # Inverse dynamics control, kinematic simulation inside MPC and forward dynamics outside controller, using
            # derived inverse dynamics
            evolve_state = ds.kinematic_evolve_state
            evolve_state_batched = True

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
            nn_dynamics=nn_dynamics,
            include_null_controls=include_null_controls,
            evolve_state_batched=evolve_state_batched
        )

    def build(self): return self.controller

    def step(self, state, ensure_control=False):
        # If in default mode, result is the direct control input, i.e. torques, otherwise is accelerations for
        # inverse dynamics control mode
        result = self.controller.step(state)

        if self.inverse_dyn_control:
            if self.nn_model is not None:
                pass
            else:
                result = self.ds.inverse_dynamics(result, state)

        if not ensure_control:
            return result

        return self.ds.ensure_control(result)
