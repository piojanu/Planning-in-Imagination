# Model-based Reinforcement Learning using different MCTS variants

Goal of this experiment is to implement [MCTSnets](https://arxiv.org/abs/1802.04697) and check their transfer learning capabilities. It was shown that training them in supervised fashion is possible, so we can use it and check on our datasets their efficiency in transferring knowledge between eleven different Atari games.  
Before MCTSnets, we will implement other MCTS based algorithms (UCT variant of MCTS with pretrained value-function for leaf states evaluation and [AlphaZero](http://tim.hibal.org/blog/alpha-zero-how-and-why-it-works/)). Those will serve as intermediate phases (AlphaZero is just extended Value-function MCTS and MCTSnets algorithm is extended AlphaZero). Also we will use AlphaZero to check its transfer learning capabilities too.  
Before working with Atari games we will test our implementations on different perfect information games and then transfer them to imperfect environments cases (using learned world models).

## Core features to include...
* Support for perfect and imperfect information models.
* Support for single-player (Atari) and two-player (board) games.
* Easy switching between:
  1. [ ] Value-function MCTS
  2. [ ] AlphaZero
  3. [ ] MCTSnets

## Repository organization

```
.
├── ...
└── MCTS_agents 
    ├── README.md   (Experiment description, organization, how to run etc.)
    ├── doc         (Articles, presentations, experiments descriptions and results etc.)
    ├── etc         (Other resources related to experiment e.g. papers, diagrams etc.)
    ├── third_party (Code, scripts, whole repos but from third party)
    └── src         (All code lives here)
        ├── games       (Environments and and world models)
        ├── logs        (All the logging related files)
        ├── nets        (NNs implementations and training)
        |   └── checkpoints    (Saved models etc.)
        ├── tree        (Classes of tree structure)
        ├── script1.py  (All scripts performing experiments live in here)
        └── ...         (All tests and modules used for experiments live in here)

```

## How to run tests?

Make sure you have `pytest` installed. Then go to `<root>/MCTS_agents/src/` dir and run `pytest`.
