# Stratified DDPG Implementation

In this method, we employed two distinct buffers derived from
the complete replay buffer. The concept involves adding episodes exceeding
a specific return threshold t to the ”good” buffer, while those falling below
are allocated to the ”bad” buffer. The threshold parameter is updated
dynamically according to the agent’s experiences, as shown in the following
equation:

t_i = μ_i + x ∗ σ_i

In this equation, the new threshold is determined by adding the mean
μ to x times the standard deviation of rewards σ from the last 50
episodes. We utilized x ∈ {−0.5, 0.5} during our experiments. This approach
aims to leverage the variability of experiences to perform upper and lower
confidence bound weighted sampling of the replay buffer. During sampling,
priority can be given either to trajectories from the good or from the bad
buffer. This allows to tune the composition of the mini-batch from which
the networks learn. We used ratios of 0.7 and 0.5 when sampling.