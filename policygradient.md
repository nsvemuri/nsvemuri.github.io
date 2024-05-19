# Policy gradient methods

<!--* freshness: { owner: 'nagasrinivas' reviewed: '2024-05-11' review_interval: '12 months'} *-->

### Objective

*   Solve Grid world problem using Policy graident Technique

### Policy Gradient Method (REINFORCE with baseline)

[Colab Link:](https://colab.research.google.com/drive/1gHW3MEemv-Ocq-9YcNm3vIFukLRRNhXd#scrollTo=en_xBj3GB-iq)

Policy gradient methods directly optimize the policy itself, as opposed to other
reinforcement learning techniques (like Q-learning) that learn value functions.
Here's the core idea:

#### Parametrized Policy:

The policy is represented by a neural network with adjustable parameters
(weights). This network takes the current state as input and outputs
probabilities for each possible action.

π(a|s, θ) where θ is the gradient

#### Sampling Trajectories:

The robot interacts with the environment by following the policy. This generates
trajectories—sequences of states, actions, and rewards.

#### Policy Gradient Estimation:

The algorithm estimates the policy gradient—the direction to adjust policy
parameters to increase the probability of actions that lead to higher rewards.
This is done using the following update rule:

Δθ = α * ∇θ log π(a|s, θ) * G_t Where:

```
Δθ: Change in policy parameters
α: Learning rate
∇θ log π(a|s, θ): Gradient of the log probability of taking action 'a' in state 's' with respect to policy parameters θ
G_t: Discounted return from time step 't' onwards
```

#### Baselines: Why They Matter

The REINFORCE algorithm can suffer from high variance in the gradient estimates.
This is where the baseline comes into play. A baseline is a function of the
state that helps reduce variance while keeping the gradient estimates unbiased.

The update rule is modified as follows:

Δθ = α * ∇θ log π(a|s, θ) * (G_t - b(s)) Where:

b(s): Baseline value for state 's'

The baseline acts as a reference point for the rewards. If a reward is higher
than the baseline, it indicates the chosen action was better than expected, and
the probability of that action is increased. If a reward is lower than the
baseline, the probability of that action is decreased.

##### Average Reward:

A simpler baseline is the average of the rewards observed so far for each state.
This provides a less accurate estimate but is computationally easier.

#### Removing Loops

For faster convergence we removed the loops during the state traversals

### [Dynamic Programming](dynamicprogramming.md)

