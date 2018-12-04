# Planning (using great AlphaZero algorithm) and learning from raw pixels observations (using World Models to model dynamics)

If World Models controller was able to train solely in imagination, then maybe representation it creates from pixels is sufficient for AlphaZero to be applied in closed environments (without access to env. dynamic).  
We want to leverage great planning power of AlphaZero in high-dim. environments where agent don't have access to game rules i.e. agent have to learn from observations. We want to use World Models to create compact representation of the environment that could be used to train dynamics model and then finally AlphaZero algorithm. More [here](https://www.evernote.com/l/AjgnLgyX35FE1bJNn6InNWcZFDGi8f5ShEg).

## Repository organization

```
.
├── README.md (This file. Organization, road map, goals etc.)
├── etc (Other resources related to project in general e.g. notes, papers, lectures)
└── <experiment/sub-project name> 
    ├── README.md (Experiment description, organization, how to run etc.)
    ├── doc (Articles, presentations, experiments descriptions and results etc.)
    ├── etc (Other resources related to experiment e.g. papers, diagrams etc.)
    └── src (All code lives here)
        ├── checkpoints (Saved models etc.)
        ├── codebase    (Classes, helpers, utils etc.)
        ├── logs        (All the logging related files)
        ├── out         (All side products of scripts that don't fit anywhere else)
        ├── third_party (As `codebase` + scripts but from third party)
        └── script1.py  (All scripts performing experiments live in `src`)

```

## Places

* [CODE_OF_CONDUCT.md](https://github.com/piojanu/Transfer-Learning-in-RL/blob/master/.github/CODE_OF_CONDUCT.md) - our rules, how to behave.
* [CONTRIBUTION.md](https://github.com/piojanu/Transfer-Learning-in-RL/blob/master/.github/CONTRIBUTING.md) - contributing guidelines, **mandatory to read and apply!**
* [LINKS.md](https://github.com/piojanu/Transfer-Learning-in-RL/blob/master/.github/LINKS.md) - useful links to articles, lectures etc. **with short description**.

## Planning in Imagination config and training

### Config

#### Memory component

To train the Memory component, run:

_To be completed..._

Config options:

```
"rnn_training": {
    "batch_size"         : 128,
    "sequence_len"       : 1000,                 -- Sequence length to use in RNN. If recorded episode is shorter, it will
                                                 -- be padded with zeros.
    "terminal_prob"      : 0.2,                  -- Probability that sampled sequence will finish with terminal state.
    "hidden_units"       : 256,                  -- Number of neurons in RNN's hidden state.
    "epochs"             : 1000,
    "learning_rate"      : 0.001,
    "patience"           : 10,                   -- After this many epochs, if validation loss does not improve, the training is stopped.
    "rend_n_rollouts"    : 10,                   -- Render N simulated steps using memory module. Can't be greater than sequence_len/2.
    "rend_n_episodes"    : 12,                   -- Render visualization for N episodes.
    "rend_step"          : 4,                    -- Render every Nth frame. rend_step*rend_n_rollouts can't be greater than sequence_len/2
    "exp_replay_size"    : 200000,               -- How many games to keep in experience replay.
    "gamma"              : 1.0                   -- Discount factor, used to calculate state values.
    "ckpt_prefix"        : "./ckpt/memory",      -- Prefix used in path to best model (checkpoint).
    "logs_dir"           : "./logs"              -- Path to directory with logs.
}
```

By default best model will be loaded from last iteration checkpoint.
