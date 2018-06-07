# Transfer Learning In Model-based Reinforcement Learning

Nowadays AI agents are highly specialized in solving specific problems. Those models are typically trained in separation, without sharing knowledge which is noneffective. Huge amount of effort is put into research aiming at transfer learning, that is using knowledge obtained in one problem when solving other related problems.  
Goal of this project is to evaluate method of transfer learning between different Atari games via generative models described [here](https://blog.openai.com/requests-for-research-2/).

## Road map
1. **Until end of April:** Study model-based reinforcement learning and generative models.
2. **Until end of June:** Proof of concept - Model-based agent (using generative model) for Atari games.
3. **Until end of July:** Transfer Learning experiments at an advanced stage, work ended or nearly ended.
4. **Until end of August:** Paper finished, applying for participation in conferences!

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

