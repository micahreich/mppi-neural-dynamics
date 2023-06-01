# Model Predictive Path Integral Control with Learned Dynamics
Final Project for 16-711 Robot Kinematics, Dynamics, and Control

### Abstract
Optimal control of robotic systems with non-linear dynamics and complex objective functions remains a challenging yet fundamental area of robotics research. We present a nonlinear Model Predictive Control (MPC) scheme for systems with unknown or complex dynamics by way of a neural dynamics model. Through the use of a Model Predictive Path Integral (MPPI) controller and sampling-based action sequence optimization, we demonstrate that neural network dynamics models can be substituted into traditional MPC schemes and achieve high performance in cost minimization. In addition, we show that neural networks can be used both within controllers as forward dynamics models as well as exterior to controllers as inverse dynamics models. 

### Model Predictive Control and MPPI
MPPI is an MPC scheme that uses Monte-Carlo sampling of controls to approximate a control sample $u \sim Q$ from the time-horizon optimal control distribution $Q$. MPPI uses $K$ independent rollouts and a time horizon, $T$ to explore the state space with sampled control sequences, choosing a sequence (or weighted average of sequences) which yield low cost trajectories. This results in a versatile scheme which does not impose particular requirements (i.e. convex costs, linear dynamics) on the  optimization problem.

### Neural Network Dynamics Models
Traditionally, MPC uses a derived dynamics model for the state transition, $\textbf{f}(\cdot)$, or for the inverse dynamics, $\textbf{f}^{-1}(\cdot)$. We demonstrate the ability to instead model $\textbf{f}(\cdot)$ and $\textbf{f}^{-1}(\cdot)$ as neural networks which, in the forward case, predict forces and torques from the system state and accelerations, and in the inverse case, predict accelerations from control forces and torques.

### Paper
For information about our methods and implementation, please see our [final paper](https://github.com/micahreich/mppi-neural-dynamics/blob/main/docs/16_711_final_paper-2.pdf).
