import sys
sys.path.append("..")

from mppi_controller import MPPIController, ControlNoiseInit
import numpy as np


class SystemController:
    def __init__(self, ds,
                 n_rollouts, horizon_length, exploration_cov, exploration_lambda,
                 alpha_mu, alpha_sigma,
                 state_cost, terminal_cost, control_cost=None,
                 nn_model=None, control_range=None, control_noise_initialization=ControlNoiseInit.RANDOM,
                 include_null_controls=False, inverse_dyn_control=False):
        
        if inverse_dyn_control:
            evolve_state = ds.kinematic_evolve_state
            evolve_state_batched = True
            nn_dynamics = False
            nn_model = None
        else:
            if nn_model is None:
                evolve_state = ds.integrator.step
                evolve_state_batched = nn_dynamics = False
            else:
                def nn_evolve_state(nn_input):
                    states, controls = nn_input[:, :ds.nx], nn_input[:, ds.nx:]
                    x_ddots = nn_model(nn_input)

                    x_dots = states[:, 1::2]

                    next_states = states
                    next_states[:, ::2] += x_dots * ds.dt + 1 / 2 * x_ddots * (ds.dt ** 2)
                    next_states[:, 1::2] += x_ddots * ds.dt

                    return next_states

                evolve_state = nn_evolve_state
                nn_dynamics = evolve_state_batched = True

        self.controller = MPPIController(
            n_rollouts=n_rollouts,
            horizon_length=horizon_length,
            exploration_cov=exploration_cov,
            exploration_lambda=exploration_lambda,
            alpha_mu=alpha_mu,
            alpha_sigma=alpha_sigma,
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

        self.n_rollouts = n_rollouts
        self.horizon_length = horizon_length
        self.exploration_cov = exploration_cov
        self.exploration_lambda = exploration_lambda
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma
        self.terminal_cost = terminal_cost
        self.state_cost = state_cost
        self.evolve_state = evolve_state
        self.control_cost = control_cost
        self.control_range = control_range
        self.control_noise_initialization = control_noise_initialization
        self.nn_model = nn_model
        self.nn_dynamics = nn_dynamics
        self.include_null_controls = include_null_controls
        self.evolve_state_batched = evolve_state_batched
        self.inverse_dyn_control = inverse_dyn_control

    def build(self):
        return self.controller

    def step(self, state, ensure_control=False, debug=False):
        """
        Wrapper for MPPI step function

        :param state: current system state
        :param ensure_control: flag to run the system's ensure_control function before returning control input
        :return: next control sequence to send to system actuators
        """
        result = self.controller.step(state, debug=debug)

        if self.inverse_dyn_control:
            # If using inverse dynamics, need to run inverse dynamics on the controller
            # acceleration output
            if self.nn_model is None:
                result = self.ds.inverse_dynamics(state, result)
            else:
                nn_input = np.hstack((state, result))
                result = self.nn_model(nn_input)

        if not ensure_control:
            return result

        return self.ds.ensure_control(result)
