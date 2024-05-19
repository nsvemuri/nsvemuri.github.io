# Monte Carlo Technique

<!--* freshness: { owner: 'nagasrinivas' reviewed: '2024-05-11' review_interval: '12 months'} *-->

### Objective

*   Solve the Grid world Navigation problem with a specific class of algorithms
    called Monte Carlo learning
    *   Monte Carlo Simulation with State values
    *   Monte Carlo Simulation with Action values

### Monte Carlo Simulation

#### Monte Carlo Simulation with State value function

[Colab Link: Monte Carlo Simulation Learning with State value functions](https://colab.research.google.com/drive/1ECb_3WfaVHWQtmWkuiCpGuV9SKwX8Wuv#scrollTo=G9-TFrJLxcb_)

Monte Carlo Simulation Learning algorithm for the Grid Navigation problem - We
learn State Value function V for num_episodes - Policy is a mapping from state
-> action - We need ***model 'P'*** of the environment to identify the policy
from the state value function - First we use the current policy to complete the
entire episode

Crux of the algorithm in State values is:

```
# Compute Monte Carlo returns
G = 0
for t in range(len(episode_states)-1, -1, -1):
    G = gamma * G + episode_rewards[t]
    state = episode_states[t]
    C[state] = C[state] + 1
    V[state] = V[state] + (G - V[state]) / (C[state])  # Update value function
```

We use episodes to learn the State value function and learn at the end of the
episode

Finally policy is learnt by picking a direction that maximises state-value
function

```
for s in range(env.nS):
    policy[s] = np.argmax([sum([p * (r + gamma * V[ns]) for p, ns, r, is_done in env.P[s][a]]) for a in range(env.nA)])
```

#### Monte Carlo Simulation with Action value methods

[Colab Link: Monte Carlo Simulation Learning with Action value functions](https://colab.research.google.com/drive/1ECb_3WfaVHWQtmWkuiCpGuV9SKwX8Wuv#scrollTo=tNAbebpXtD1k)

Not always, we have the model of the environment, in such cases, we learn the
Action Value method

Monte Carlo Simulation Learning is a action-value based Learning technique for
identifying the optimal policy

*   First entire episode is run using the current epsilon greedy policy
*   Update action avalues using the rewards observed

CRUX of the algorithm is below:

```
G = 0
for t in range(len(episode_states)-1, -1, -1):
    G = gamma * G + episode_rewards[t]
    state = episode_states[t]
    if state not in episode_states[0:t-1]:
      action = episode_actions[t]
      C[state, action] = C[state, action] + 1
      Q[state, action] = Q[state, action] + (G - Q[state,action]) / C[state,action]  # Update Q-value
```

It is expected to have high variance compared to the TD Learning methods.
However it has lower bias compared to TD methods.

### [Policy gradient methods](policygradient.md)

