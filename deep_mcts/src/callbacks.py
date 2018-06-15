import logging as log
import numpy as np
import os
import sys

from collections import deque
from humblerl import Callback
from pickle import Pickler, Unpickler


class BasicStats(Callback):
    def __init__(self):
        self.steps = [0, ]
        self.returns = [0, ]

    def on_step_taken(self, transition):
        self.steps[-1] += 1
        self.returns[-1] += transition.reward

    def on_episode_end(self):
        logs = {}
        logs["# steps"] = self.steps[-1]
        logs["max # steps"] = np.max(self.steps)
        logs["min # steps"] = np.min(self.steps)
        logs["avg # steps"] = np.mean(self.steps)

        logs["return"] = self.returns[-1]
        logs["max return"] = np.max(self.returns)
        logs["min return"] = np.min(self.returns)
        logs["avg return"] = np.mean(self.returns)

        self.steps.append(0)
        self.returns.append(0)

        return logs

    def on_loop_finish(self, is_aborted):
        self.steps = [0, ]
        self.returns = [0, ]


class Storage(Callback):
    def __init__(self, params={}):
        """
        Storage with train examples.

        Args:
            params (JSON Dictionary):
                * 'exp_replay_size' (int)   : Max size of big bag. When big bag is full then oldest
                                              element is removed. (Default: 100000)
                * 'store_dir' (string)      : Directory where to store/load from big bag.
                                              (Default: "checkpoints")
                * 'store_filename' (string) : Filename of stored data file.
                                              (Default: "data.examples")
        """

        self.params = params
        self.small_bag = deque()
        self.big_bag = deque()

        self._recent_action_probs = None

    def on_action_planned(self, logits):
        self._recent_action_probs = logits / np.sum(logits)

    def on_step_taken(self, transition):
        small_package = transition.state, self._recent_action_probs, transition.reward
        self.small_bag.append(small_package)
        if len(self.small_bag) >= self.params.get("exp_replay_size", 100000):
            self.small_bag.popleft()

        if transition.is_terminal:
            for package in self.small_bag:
                big_package = package[0], package[1], transition.reward
                if len(self.big_bag) >= self.params.get("exp_replay_size", 100000):
                    self.big_bag.popleft()
                self.big_bag.append(big_package)

    def on_episode_end(self):
        logs = {"# samples": len(self.big_bag)}
        self.small_bag.clear()

        return logs

    def store(self):
        folder = self.params.get("store_dir", "checkpoints")
        if not os.path.exists(folder):
            log.warn("Examples store directory does not exist! Creating directory {}".format(folder))
            os.makedirs(folder)

        filename = self.params.get("store_filename", "data.examples")
        path = os.path.join(folder, filename)
        with open(path, "wb+") as f:
            Pickler(f).dump(self.big_bag)

    def load(self):
        folder = self.params.get("store_dir", "checkpoints")
        filename = self.params.get("store_filename", "data.examples")
        path = os.path.join(folder, filename)
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

    def on_step_taken(self, transition):
        if transition.is_terminal:
            if transition.reward == 0:
                self.draws += 1
            elif transition.reward > 0:
                self.wins += 1
            else:
                self.losses += 1

    def on_episode_end(self):
        return {"P1 wins": self.wins, "P2 wins": self.losses, "draws": self.draws}

    def reset(self):
        self.wins, self.losses, self.draws = 0, 0, 0
