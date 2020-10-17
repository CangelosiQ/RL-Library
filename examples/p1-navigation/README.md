[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation
## 1. Project description and setup (copied from [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md))
### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in this repository, in the `examples/navigation/` folder, and unzip (or decompress) the file. 

## 2. Project Submission
In the following section, I will provide information going through the list of [criteria](https://review.udacity.com/#!/rubrics/1889/view) for the project.

### Training Code
The code for this project is located under `examples/navigation`.
To train the agent you can run either `Navigation.ipynb` or `Navigation.py`. My working environment for this project was the python script while the python notebook was used to describe the code and its outputs to the reviewers.


**Framework**: The code is written in PyTorch and Python 3.

**Saved Model Weights**: The model weights are saved under `examples/navigation/DDQN_20_15_8/checkpoint.pth` and can be easily loaded when running `Navigation.ipynb` or `Navigation.py` with `mode="test"`

### README
This is exactly the file currently being read :-) describing the project environment details, providing instructions for installing dependencies and on how to run the code in the repository, to train the agent.

### Report
The report for this project is `examples/navigation/Report.md`.