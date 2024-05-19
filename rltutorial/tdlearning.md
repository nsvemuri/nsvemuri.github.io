# TD Learning

<!--* freshness: { owner: 'nagasrinivas' reviewed: '2024-05-11' review_interval: '12 months'} *-->

### Objective

*   Solve the Grid world Navigation problem with a specific class of algorithms
    called TD learning
    *   TD(0)
    *   TD(n) step Bootstrapping
*   Understanding concepts:
    *   Prediction vs Control methods
    *   State value methods vs Action value methods

### TD(0)

[Colab Link: TD(0) Learning with State value functions](https://colab.research.google.com/drive/1pROz-PoAMjv8xERnocSqmFj8MAWHqcpx#scrollTo=G9-TFrJLxcb_)

#### TD(0) with State value function

*   We learn State Value function V[env.nS] for num_episodes iterations
*   Policy is a mapping from state -> action
*   We need **model 'P'** of the environment to identify the policy from the
    learnt state value function
*   We update relevant state values at every step of the episode

#### TD(0) one step Update

```python
td_target = reward + gamma * V[next_state]
td_error = td_target - V[state]
V[state] += alpha * td_error
```

#### TD(0) Deriving policy from model + state value function

In simple words, it is the argmax of action leading to the adjacent state-values
+ reward to reach that state

```python
# policy = np.zeros(env.nS, dtype=int) ....
# Get optimal policy from learned value function
    for s in range(env.nS):
        policy[s] = np.argmax([sum([p * (r + gamma * V[ns]) for p, ns, r, is_done in env.P[s][a]]) for a in range(env.nA)])
```

#### Theory: Prediction vs Control

##### Policy Evaluation: ~ Prediction Problem

Definition: The process of determining the value function V(s) for each state
's' under a fixed policy π. In essence, it answers the question: "How good is it
to be in a particular state if we follow this policy?"

##### Policy Improvement

Definition: The step of generating a new policy π' that selects actions greedily
based on the estimated value function from the policy evaluation step. The new
policy aims to be better than the previous one by always choosing the action
with the highest estimated value in each state.

##### Control Problem

Definition: The control problem is the central challenge of reinforcement
learning. It aims to find an optimal policy that maximizes the expected
cumulative reward an agent can achieve. Key Question: "What is the best policy
to follow in order to maximize rewards?"

###### Policy Iteration: It is a specific technique to solve the control problem.

We just did policy iteration in colab to solve the Grid navigation problem

### Action value methods

Not always, we have the model of the environment, in such cases, we learn the
Action Value method

#### Q Learning

[Colab Link: TD(0) Learning with Q Learning](https://colab.research.google.com/drive/1pROz-PoAMjv8xERnocSqmFj8MAWHqcpx#scrollTo=deTbX8-3RvO3)

Q Learning is a action-value based TD(0) Learning technique for identifying the
optimal policy

CRUX of the algorithm is below:

```
    # Q-learning update
    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
```

It is also known as off-policy technique because Update step above doesn't
strictly use the behavioural policy in its update (it uses np.max() instead)

#### SARSA

[Colab Link: TD(0) Learning with SARSA](https://colab.research.google.com/drive/1pROz-PoAMjv8xERnocSqmFj8MAWHqcpx#scrollTo=sSImSlHqT6kV)

SARSA Learning is a action-value based TD(0) Learning technique for identifying
the optimal policy

CRUX of the algorithm is below:

```
    # SARSA update
    Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

    state = next_state
    action = next_action

    # Get optimal policy from learned action-values
    policy = np.argmax(Q, axis=1)
```

It is also known as ON-policy technique because Update step above strictly uses
the behavioural policy in its update

#### TD N-step bootstrapping

[Colab Link: TD(n) bootstrap Learning](https://colab.research.google.com/drive/1pROz-PoAMjv8xERnocSqmFj8MAWHqcpx#scrollTo=OyPP-SrzWi_W)

TD(n) Bootstrapping Learning algorithm for the Grid Navigation problem

*   We learn Action Value function Q for num_episodes
*   After every step of the episode, we look back 'n' steps to compute G (+
    another increment if the episode didn't end)
*   Update action-value based on TD delta

Crux of the algorithm is:

```
      G = 0
      for t in range(len(states)-1, len(states)-1-n, -1):
        if (len(states) >= n):
          G = gamma * G + rewards[t]
          constant = constant * gamma

      if not done and len(states) >= n:
        G = G + constant * Q [next_states[len(states)-1], policy[next_states[len(states)-1]]]

      # TD update
      Q[states[len(states)-1-n], actions[len(states)-1-n]] += alpha * (G - Q[states[len(states)-1-n], actions[len(states)-1-n]])
```

### Next [Monte Carlo simulation methods](montecarlo.md)

