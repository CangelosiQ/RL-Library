# Report for Project 2: Continuous Control
Author: Quentin Cangelosi, <cangelosi.quentin@gmail.com>


This report details the methods, algorithms, results and learnings from working on the Continuous Control project, as part of the Udacity Nanodegree Deep Reinforcement Learning. 


![Alt Text](DRL-Navigation.gif)



## Source Code

As for the first project, the source code to solve this second project is wrapped up in the "rl_library", aiming at creating the foundation of generalisable, reusable and maintainable of Deep Reinforcement Learning algorithms. 

The entry-points to solve this project are the notebook and script `examples/p2_continuous-control/Continuous_Control(.py/.ipynb)`. The python script being the version used during development and the notebook being a replicate allowing reviewers to visualize the outputs.  

The DDPG agent is implemented under `rl_library/agents/ddpg_agent.py` and is based on the example provided in the nanodegree to solve the Bipedal-Walker environment. The pytorch models for the Actor and the Critic networks are instanciated from the scripts `rl_library/agents/models/heads.py` and `rl_library/agents/models/body.py`. Various utils used by the DDPG agent are located under `rl_library/utils` as the normalizers, replay buffer, noise classes and visualization methods.



## Methods and Algorithms

### Method

First, I started to work on the DDPG agent using the example of the Bipedal-Walker, which was not really solved with the code and hyper-parameters provided in the nanodegree. From there my first initiative was to move this code in the rl_library, improved the monitors already started and continued from there. As recommended by the teacher Miguel Morales, I perused the [DeepRL](https://github.com/ShangtongZhang/DeepRL) repository from  ShangtongZhang. I found the general implementation of the neural networks into head and body classes very elegant and reused this concept. I also read the [DDPG paper](https://arxiv.org/abs/1509.02971) several time. Unfortunately, I spent enormous amount of time trying to solve the Bipedal-Walker-v3 and the single agent continuous-control Unity environment with not much success at first. I implemented different noise options to add to the actions or to the network weights, different batch and reward normalizations, added learning rate schedulers for the network optimizers and did a lot of hyperparameter tuning which I could analyse thanks to thourough book-keeping of every parameter into configuration JSON files. After several weeks into this process, I decided to review how others solved this continuous-control problem and could finally spot a mistake I made early on when splitting actor and critics neural networks into heads and bodies: the critic last layer was not a single value for Q(s, a) but had the same size as the actor last layer, one node per action, and this was not breaking anything nor leading to noticeable divergence. It was not a pleasant experience for sure but after understanding my mistake and looking back at all the work I had done to try improving the DDPG agent gave me a much deeper look at the algorithm and various tricks around it.  



## Results Comparison


## Further Work


## Self-Evaluation against evaluation criteria of the report


#### Report
    The submission includes a file in the root of the GitHub repository or zip file 
    (one of Report.md, Report.ipynb, or Report.pdf) that provides a description of 
    the implementation.

Yes, this is the document being read.


#### Learning Algorithm
    The report clearly describes the learning algorithm, along with the chosen hyperparameters. 
    It also describes the model architectures for any neural networks.

Please read section Methods and Algorithms.


#### Plot of Rewards
    A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. The submission reports the number of episodes needed to solve the environment.


#### Ideas for Future Work
    The submission has concrete future ideas for improving the agent's performance.
        
Please read section Futher Work.