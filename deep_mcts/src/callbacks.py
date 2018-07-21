import csv
import logging as log
import numpy as np
import os
import sys

from collections import deque
from humblerl import Callback
from pickle import Pickler, Unpickler


class CSVSaverWrapper(Callback):
    """Saves to .csv file whatever source callback logs.

    Args:
        callback (Callback): Source callback to save logs from.
        path (string): Where to save logs.
        only_last (bool): If only save last log in the loop. Useful when source
    callback aggregates logs. (Default: False)
    """

    def __init__(self, callback, path, only_last=False):
        self.callback = callback
        self.path = path
        self.only_last = only_last
        self.history = []

    def on_loop_start(self):
        return self.callback.on_loop_start()

    def on_action_planned(self, logits, metrics):
        return self.callback.on_action_planned(logits, metrics)

    def on_step_taken(self, transition):
        return self.callback.on_step_taken(transition)

    def on_episode_end(self):
        logs = self.callback.on_episode_end()
        self.history.append(logs)
        return logs

    def on_loop_finish(self, is_aborted):
        self._store()
        return self.callback.on_loop_finish(is_aborted)

    def _store(self):
        if not os.path.isfile(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)

            with open(self.path, "w") as f:
                logs_file = csv.DictWriter(f, self.history[0].keys())
                logs_file.writeheader()

        with open(self.path, "a") as f:
            logs_file = csv.DictWriter(f, self.history[0].keys())
            if self.only_last:
                logs_file.writerow(self.history[-1])
            else:
                logs_file.writerows(self.history)
            self.history = []


class BasicStats(Callback):
    """Gather basic episode statistics like:
      * number of steps,
      * return,
      * max reward,
      * min reward.
    """

    def __init__(self, save_path=None):
        self._reset()

    def on_step_taken(self, transition):
        self.steps += 1
        self.rewards.append(transition.reward)

    def on_episode_end(self):
        logs = {}
        logs["# steps"] = self.steps
        logs["return"] = np.sum(self.rewards)
        logs["max reward"] = np.max(self.rewards)
        logs["min reward"] = np.min(self.rewards)

        self._reset()
        return logs

    def on_loop_finish(self, is_aborted):
        self._reset()

    def _reset(self):
        self.steps = 0
        self.rewards = []


class Storage(Callback):
    def __init__(self, model, params={}):
        """
        Storage with train examples.

        Args:
            params (JSON Dictionary):
                * 'exp_replay_size' (int)   : Max size of big bag. When big bag is full then oldest
                                              element is removed. (Default: 100000)
                * 'save_data_path' (string) : Path to file where to store/load from big bag.
                                              (Default: "./checkpoints/data.examples")
        """

        self.model = model
        self.params = params
        self.small_bag = deque()
        self.big_bag = deque()

        self._recent_action_probs = None

    def on_action_planned(self, logits, metrics):
        # Proportional without temperature
        self._recent_action_probs = logits / np.sum(logits)

    def on_step_taken(self, transition):
        # NOTE: We never pass terminal state (it would be next_state), so NN can't learn directly
        # what is the value of terminal/end state.
        small_package = transition.player, transition.state, self._recent_action_probs
        self.small_bag.append(small_package)
        if len(self.small_bag) > self.params.get("exp_replay_size", 100000):
            self.small_bag.popleft()

        if transition.is_terminal:
            for player, state, mcts_pi in self.small_bag:
                cannonical_state = self.model.get_canonical_form(state, player)
                # Reward from env is from player one perspective,
                # so we check if current player is the first one
                player_reward = transition.reward * (1 if player == 0 else -1)
                self.big_bag.append((cannonical_state, mcts_pi, player_reward))
                if len(self.big_bag) > self.params.get("exp_replay_size", 100000):
                    self.big_bag.popleft()
            self.small_bag.clear()

    def on_episode_end(self):
        logs = {"# samples": len(self.big_bag)}
        return logs

    def store(self):
        path = self.params.get("save_data_path", "./checkpoints/data.examples")
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            log.warn("Examples store directory does not exist! Creating directory {}".format(folder))
            os.makedirs(folder)

        with open(path, "wb+") as f:
            Pickler(f).dump(self.big_bag)

    def load(self):
        path = self.params.get("save_data_path", "./checkpoints/data.examples")
        if not os.path.isfile(path):
            r = input("File with train examples was not found. Continue? [y|n]: ")
            if r != "y":
                sys.exit()
        else:
            log.info("File with train examples found. Reading it.")
            with open(path, "rb") as f:
                self.big_bag = Unpickler(f).load()

            # Prune dataset if too big
            while len(self.big_bag) > self.params.get("exp_replay_size", 100000):
                self.big_bag.popleft()


class Tournament(Callback):
    def __init__(self):
        self.reset()

    @property
    def results(self):
        return self.wins, self.losses, self.draws

    def on_loop_start(self):
        self.reset()

    def on_step_taken(self, transition):
        if transition.is_terminal:
            # NOTE: Because players have fixed player id, and reward is returned from
            # perspective of player one, we are indifferent to who is starting the game.
            if transition.reward == 0:
                self.draws += 1
            elif transition.reward > 0:
                self.wins += 1
            else:
                self.losses += 1

    def on_episode_end(self):
        return {"wannabe": self.wins, "best": self.losses, "draws": self.draws}

    def reset(self):
        self.wins, self.losses, self.draws = 0, 0, 0


class RenderCallback(Callback):
    def __init__(self, env, render):
        self.env = env
        self.render = render

    def on_step_taken(self, _):
        self.do_render()

    def on_episode_end(self):
        return {}

    def do_render(self):
        if self.render:
            self.env.render()
