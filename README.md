# DDPG - Teaching a Robot How to Walk
- Authors: Jules Kreuer, Roman Machacek, Jonas Müller
- Date of submission: May 08, 2023

## Introduction

This project aims to implement and improve upon the Deep Deterministic Policy Gradient (DDPG) algorithm for continuous control tasks in robotics and games, as proposed by Lillicrap et al. [1]. DDPG is an actor-critic method that utilizes two deep neural networks for learning the actor policy and critic Q-function. The method is model-free, off-policy, and can handle high-dimensional observation spaces.

The effectiveness of DDPG is demonstrated on a variety of simulated robotics tasks, including block stacking, cartpole balancing, and humanoid locomotion. The performance of DDPG is also compared to other state-of-the-art methods, with results showing superior learning speed and final performance.

The project also explores two new contributions aimed at improving reward gains in the optimization process of the reinforcement learning pipeline.

## Repository Structure

- `basic_ddpg/`: Contains the main source code for the DDPG algorithm.
- `softmax_ddpg/`: Softmax version of DDPG algorithm.
- `stratified_ddpg/`: Stratified Buffer version of DDPG algorithm.
- `runs/`: Contains the runs of the simulated robotics environments used for testing the algorithm.
- `report/`: Contains the latex source of the report
- `Kreuer_Machacek_Mueller_DDPG_2023.pdf`: Final report
- `Lecture.pdf` Lecture / Presentation given on that topic.

## Getting Started

1. Clone the repository.
2. Install the required dependencies:
    ```
    gymnasium=0.27.1
    imageio=2.27.0
    imageio-ffmpeg=0.4.8
    matplotlib=3.2.2
    natsort=8.3.1
    numpy=1.24.2
    openjpeg=2.5.0
    pandas=1.5.3
    pillow=9.4.0
    pygame=2.1.3.dev8
    python=3.8.16
    pytorch=1.7.1
    ```
3. Run the experiments with different configurations:
    ```
    python test_bench.py
    ```
4. Analyze and visualize the results saved in `runs/`
5. Plot the average loss using `plot_average_loss.py`

## Results and Discussion

Our implementation explores two strategies aimed at enhancing the performance of the DDPG algorithm: the stratification approach and the softmax method. However, the obtained results are inconclusive, and we encountered several challenges during the implementation and evaluation of these methods.

The stratification approach carries the risk of the algorithm falling into local maxima, depending on the environment's characteristics. On the other hand, the softmax implementation proved to be computationally heavy and limited our ability to explore different hyperparameters. Furthermore, both methods require hyperparameter optimization, which adds complexity to the implementation.

Despite the inconclusive results, this project still demonstrates the utility of the DDPG algorithm for solving continuous control problems with reinforcement learning. Future work could investigate alternative methods for improving the replay buffer sampling process or explore more complex environments to test the proposed enhancements.

## References

[1] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, et al. "Continuous control with deep reinforcement learning". In: arXiv preprint arXiv:1509.02971 (2015).
