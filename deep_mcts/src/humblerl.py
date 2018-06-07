import numpy as np

from abc import ABCMeta, abstractmethod
from collections import namedtuple

Transition = namedtuple(
    "Transition", ["player", "state", "action", "reward", "next_state", "is_terminal"])


def ply(env, mind, player=0, policy='deterministic', vision=Vision(), **kwargs):
    """Conduct single ply (turn taken by one of the players).

    Args:
        env (Environment): Environment to take actions in.
        mind (Mind): Mind to use while deciding on action to take in the env.
        player (int): Player index which ply this is. (Default: 0)
        policy (string: Describes the way of choosing action from mind predictions (see Note).
        vision (Vision): State and reward preprocessing. (Default: no preprocessing)
        **kwargs: Other keyword arguments may be needed depending on chosen policy.

    Return:
        Transition: Describes transition that took place. It contains:
          * 'player'       : player index which ply this is (zero is first),
          * 'state'        : state from which transition has started (it's preprocessed with Vision),
          * 'action'       : action taken (chosen by policy),
          * 'reward'       : reward obtained (preprocessed with Vision),
          * 'next_state'   : next state observed after transition (it's preprocessed with Vision),
          * 'is_terminal'  : flag indication if this is terminal transition (episode end).
        object: Meta information obtained from Mind.

    Note:
        Possible `policy` values are:
          * 'deterministic': default,
          * 'stochastic'   : pass extra kwarg 'temperature' otherwise it's set to 1.,
          * 'egreedy'      : pass extra kwarg 'epsilon' otherwise it's set to 0.5,
          * 'identity'     : forward whatever come from Mind.
    """

    # Get and preprocess current state
    curr_state = vision(env.current_state)

    # Infer what to do
    logits, info = mind.predict(curr_state)

    # Get action
    if policy == 'deterministic':
        action = np.argmax(logits)
    elif policy == 'stochastic':
        temp = kwargs.get('temperature', d=1.)

        # Softmax with temperature
        exps = np.exp((logits - np.max(logits)) / temp)
        probs = exps / np.sum(exps)

        # Sample from created distribution
        action = np.random.choice(len(probs), p=probs)
    elif policy == 'egreedy':
        eps = kwargs.get('epsilon', d=0.5)

        # With probability of epsilon...
        if np.random.rand(1) < eps:
            # ...sample random action, otherwise
            action = np.random.randint(len(logits))
        else:
            # ...choose action greedily
            action = np.argmax(logits)
    elif policy == 'identity':
        action = logits
    else:
        raise ValueError("Undefined policy")

    # Take chosen action
    raw_next_state, raw_reward, done = env.step(action)

    # Preprocess data and save in transition
    next_state, reward = vision(raw_next_state, raw_reward)
    transition = Transition(player, curr_state, action, reward, next_state, done)

    return transition, info


def loop(env, minds, n_episodes=1, max_steps=-1, policy='deterministic', vision=Vision(), callbacks=[], **kwargs):
    """Conduct series of plies (turns taken by each player in order).

    Args:
        env (Environment): Environment to take actions in.
        minds (Mind or list of Minds): Minds to use while deciding on action to take in the env.
    If more then one, then each will be used one by one starting form index 0.
        n_episodes (int): Number of episodes to play. (Default: 1)
        max_steps (int): Maximum number of steps in episode. No limit when -1. (Default: -1)
        policy (string: Describes the way of choosing action from mind predictions (see Note).
        vision (Vision): State and reward preprocessing. (Default: no preprocessing)
        callbacks (list of functions): Functions that take two arguments: Transition and object,
    values returned from `ply` (look there for more info). All of them will be called after each
    ply (Default: [])
        **kwargs: Other keyword arguments may be needed depending on chosen policy.

    Note:
        Possible `policy` values are:
          * 'deterministic': default,
          * 'stochastic'   : pass extra kwarg 'temperature' otherwise it's set to 1.,
          * 'egreedy'      : pass extra kwarg 'epsilon' otherwise it's set to 0.5,
          * 'identity'     : forward whatever come from Mind.
    """

    # Play given number of episodes
    for _ in range(n_episodes):
        step = 0

        # Play until episode ends or max_steps limit reached
        while max_steps == -1 or step <= max_steps:
            # Determine player index and mind
            if isinstance(minds, (list, tuple)):
                player = step % len(minds)
                mind = minds[player]
            else:
                player = 0
                mind = minds

            # Conduct ply
            transition, info = ply(env, mind, player, policy, vision, **kwargs)

            # Call callbacks and increment step counter
            for func in callbacks:
                func(transition, info)
            step += 1

            # Finish if this transition was terminal
            if transition.is_terminal:
                break


class Environment(metaclass=ABCMeta):
    """Abstract class for environments."""

    @abstractmethod
    def reset(self, train_mode=True):
        """Reset environment and return a first state.

        Args:
            train_mode (bool): Informs environment if it's training or evaluation
        mode. E.g. in train mode graphics could not be rendered. (default: True)

        Returns:
            np.Array: The initial state. 

        Note:
            In child class you MUST set `self._curr_state` to returned initial state.
        """

        # TODO (pj): Is there a better way to set `self._curr_state` then telling user
        #            to do this manually?
        pass

    @abstractmethod
    def step(self, action):
        """Perform action in environment.

        Args:
            action (list of floats): Action to perform. In discrete action space it's single
        element list with action number. In continuous case, it's action vector.

        Returns:
            np.Array: New state.
            float: Next reward.
            bool: Flag indicating if episode has ended.

        Note:
            In child class you MUST set `self._curr_state` to returned new state.
        """

        # TODO (pj): Is there a better way to set `self._curr_state` then telling user
        #            to do this manually?
        pass

    @property
    def current_state(self):
        """Access state.

        Returns:
            np.array: Current environment state.
        """

        return self._curr_state


class Mind(metaclass=ABCMeta):
    """Artificial mind of Agent."""

    @abstractmethod
    def plan(self, state):
        """Do forward pass through agent model, inference/planning on state.

        Args:
            state (numpy.Array): State of game to inference on.

        Returns:
            numpy.Array: Inference result, depends on specific model
        (e.g. action unnormalized log probabilities/logits).
            object: Meta information which can be accessed later with transition.
        """

        pass


class Vision(object):
    """Vision system entity in Reinforcement Learning task.

       It is responsible for data preprocessing.
    """

    def __init__(self, state_processor_fn=None, reward_processor_fn=None):
        """Initialize vision processors.

        Args:
            state_processor_fn (function): Function for state processing. It should
        take raw environment state as an input and return processed state.
        (Default: None which will result in passing raw state)
            reward_processor_fn (function): Function for reward processing. It should
        take raw environment reward as an input and return processed reward.
        (Default: None which will result in passing raw reward)
        """

        self._process_state = \
            state_processor if state_processor is not None else lambda x: x
        self._process_reward = \
            reward_processor if reward_processor is not None else lambda x: x

    def __call__(self, state, reward=0.):
        return self._process_state(state), self._process_reward(reward)
