# RL-Library: a deep Reinforcement Learning Library
Author: Quentin Cangelosi, <cangelosi.quentin@gmail.com>


## About this Repository

Under the framework of the Udacity nanodegreee [Deep Reinforcement Learning](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), I decided to construct a reinforcement learning library, aiming not only at learning the courses and solving the coding exercises, but more importantly at setting up a basis library that can be reused, maintained and extended in the future. 

**Important Note**: As the decision to build a library was made in the middle of the second chapter of the course, the first algorithms studied (Monte Carlo, TD Control, etc.) have not yet been included in this repository. 

### Repository Structure

    ├── README.md           The top-level README for reviewers and developers
    │
    ├── docs                [WIP] To create a sphynx documentation of the library
    │
    ├── examples            Examples on how to use the library 
    │   ├── navigation      Project Navigation of the Udacity Nanodegree (Project 1, Part 2)
    │   └── p2_continuous-control   Project 2 of the Udacity Nanodegree 
    │
    ├── rl_library          The Deep Reinforcement Learning Library
    │   ├── agents          Reinforcement Learning Agents
    │   │   └── models      Pytorch models
    │   ├── monitors        Monitors
    │   └── utils           Utils
    │
    ├── setup_unityagents   Required to set up the UnityAgent environments to run the examples
    │ 
    ├── tests               [WIP] Unit Tests
    │
    ├── requirements.txt    Package dependencies
    │
    ├── requirements_dev.txt    Package dependencies for development
    │
    └── setup.py            Python package setup file
    
    WIP = Work In Progress

### First Steps

In order to run the examples with UnityAgents, please install the dependencies located under *setup_unityagents* by running the following commands:

    cd setup_unityagents
    pip install .

Then install the package dependencies by running **after going back to the root of the repository** (e.g. by running `cd ..`)

    pip install -r requirements.txt

Finally, install the rl_library package by running

    pip install .
or for development mode:
    
    pip install -e .
    

# Project 1: Navigation
cf. README [here](examples/p1-navigation/README.md)


# Project 2: Continuous Control
cf. README [here](./examples/p2_continuous-control/README.md)


# Project 3: Collaboration and Competition

cf. README [here](./examples/p3_collab-compet/README.md)

