
# Project III: Collaboration and Competition

## Introduction

The project uses Deep Reinforcement Learning algorithm, MADDPG, to train two agents to play Tennis game.

![Multi-Agent Reinforcement Learning - Amit Patel - Medium](https://miro.medium.com/max/1200/1*UmQHDskrYnONpVFk-1TjZA.gif)

## Project Details

#### State-Space

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

#### Rewards and Completion

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.


The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

-   After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
-   This yields a single  **score**  for each episode.

The environment is considered solved, when the average (over 100 episodes) of those  **scores**  is at least +0.5.
## Getting Started & Instructions

  

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

-   Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
-   Mac OSX:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
-   Windows (32-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
-   Windows (64-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in the in the root folder, and unzip (or decompress) the file.

3. The crucial dependencies used in the project are:

- unityagents: instructions to install: https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md
  
- NumPy: Numerical Python library, pip installation command: 'pip install numpy'

- Matplotlib: Python's basic plotting library, pip installation command: 'pip install matplotlib'

- Torch: Machine and Deep Learning library for Python, pip installation command: 'pip install torch'

The code takes form of a Jupyter Notebook. To train the agent, just run each cell one by one starting from the top.

The weights of actor and critic are saved in appropriate files.