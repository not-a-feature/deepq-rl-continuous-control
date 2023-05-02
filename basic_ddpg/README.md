# Basic DDPG Implementation

The paper "Continuous control with deep reinforcement learning" by Timothy P. Lillicrap et al. (2015) presents a model-free, online, off-policy reinforcement learning method called Deep Deterministic Policy Gradient (DDPG). The method is designed to learn control policies for continuous action spaces directly from high-dimensional sensory inputs using deep neural networks. DDPG combines the strengths of Deep Q-Networks (DQN) and deterministic policy gradient algorithms.

The code provided (ddpg.py) is an implementation of the DDPG algorithm using Python and the PyTorch library. It consists of the following components:

1. Actor and Critic networks: These are the neural networks used to approximate the policy (actor) and the action-value function (critic). They are implemented as PyTorch classes with a forward method to compute their respective outputs.
2. Ornstein-Uhlenbeck Noise: This is an exploration strategy that adds temporally correlated noise to the actions produced by the actor network. It helps explore the action space more effectively than uncorrelated noise.
3. Replay Buffer: This is a data structure that stores past experiences (state, action, reward, next state, done) and allows the algorithm to sample random minibatches for training. This helps break correlations in the observation sequence and improves learning stability.
4. Soft Update: This function updates the target networks (target_actor and target_critic) by slowly tracking the learned networks (actor and critic). This is done to improve stability during training.
5. DDPG Algorithm: This function implements the main DDPG algorithm that trains the actor and critic networks using the collected experience. It includes action selection, execution, storing experience in the replay buffer, sampling minibatches, updating the networks, and updating the target networks.

The algorithm learns the optimal policy by interacting with the environment, receiving feedback (rewards), and adjusting the actor and critic networks accordingly.


# Pedulum
The code sets up an environment (Pendulum-v1) using the Farama Foundation / OpenAI Gymnasium library and runs the DDPG algorithm for a 200 episodes to learn a control policy for the environment. The Pendulum-v1 environment is a classic control problem where the goal is to swing up a pendulum and balance it in the upright position.

The model of every 10th training episode is saved and animated.

<img src="../runs/Pendulum-v1/basic/pendulum.gif" width=200>