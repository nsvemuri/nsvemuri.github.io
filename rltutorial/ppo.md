# PPO - Proximal Policy Optimization
<!--* freshness: { owner: 'nagasrinivas' reviewed: '2024-05-11' review_interval: '12 months'} *-->

### Objective

*   Solve Grid world problem using a variant of PPO - a Policy graident Technique

### Proximal Policy Optimization

[Colab Link:](https://colab.research.google.com/drive/1gsrdJ9FfILZj35u5HwSpM5sL-QwMAkM2#scrollTo=i7tJQPedBGW3&line=3&uniqifier=1)

Policy gradient methods directly optimize the policy itself, as opposed to other
reinforcement learning techniques (like Q-learning) that learn value functions.

I did my own variation of PPO Algorithm implemented for the GridWorld problem. 

* I avoided Critic Network in favor of Average Reward
* I used very small batches
* I didn't leverage entropy in the loss function

PPO is invented at OpenAI and has become the default reinforcement learning algorithm at OpenAI because of its ease of use and good performance.

* References:https://openai.com/index/openai-baselines-ppo/
* Youtube Reference: https://www.youtube.com/watch?v=5P7I-xPq8u8&t=1s  

<iframe width="560" height="315" src="https://www.youtube.com/embed/5P7I-xPq8u8"
frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

##### Average Reward:

A simpler baseline is the average of the rewards observed so far for each state.
This provides a less accurate estimate but is computationally easier.

#### Removing Loops

For faster convergence we removed the loops during the state traversals


