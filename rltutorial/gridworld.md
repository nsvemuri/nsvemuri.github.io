# Gridworld Navigation Problem


### Objective

*   Explain RL concepts with Grid world Navigation problem.
*   The objective is to find the shortest path to a terminal state from any
    seeded starting position

### Grid World

```
Grid World environment from Sutton's Reinforcement Learning book chapter 4.
You are an agent on an MxN grid and your goal is to reach the terminal state at the top left or the bottom right corner.

For example, a 4x4 grid looks as follows:

T  o  o  o
o  x  o  o
o  o  o  o
o  o  o  T

x is your position and T are the two terminal states.

You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
Actions going off the edge leave you in your current state.
You receive a reward of -1 at each step until you reach a terminal state where a high reward is received.
```

### RL definitions

*   Every position in the above grid represents a possible state that an agent
    can end up during the episode
*   An agent traversed in a loop if the agent revisits a state that is
    encountered earlier
*   An episode consists of a complete run of actions agent took, states visited
    until it reaches a terminal state
*   An episode ends when the agent reaches the terminal state
*   Specific rewards are received from the environment when agent takes actions
    from the given states

Grid world navigation is a good representative of a subset of RL problems and it
helps the reader develop good intuition around RL.

### Model vs Model-free algorithms

*   In the grid example, assuming one step traversal at a time and not
    traversing across diagonals, state transitions are very clear depending on
    the directional action taken. Also reward is -1 except after reaching a
    terminal state. By knowing clear state transitions and rewards, we know the
    model of the environment
*   Consider a Blackjack game, the 'hit' or 'stick' action doesn't precisely
    define the next state or the reward received. (Page 93 of Chapter 5). This
    is a good example of where the model is unclear

### [TD Learning methods](tdlearning.md)

