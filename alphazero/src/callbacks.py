import logging as log
import numpy as np
import os

from collections import deque
from humblerl import Callback
from pickle import Pickler, Unpickler


class Storage(Callback):
    def __init__(self, config):
        """
        Storage with train examples.

        Args:
            config (Config): Configuration object with parameters from .json file.
        """

        self.small_bag = deque()
        self.big_bag = deque()

        self.exp_replay_size = config.storage["exp_replay_size"]
        self.save_data_path = config.storage["save_data_path"]
        self.gamma = config.planner["gamma"]

        self._recent_action_probs = None

    def on_action_planned(self, step, logits, info):
        # Proportional without temperature
        self._recent_action_probs = logits / np.sum(logits)

    def on_step_taken(self, step, transition, info):
        # NOTE: We never pass terminal state (it would be next_state), so NN can't learn directly
        #       what is the value of terminal/end state.
        self.small_bag.append(self._create_small_package(transition))
        if len(self.small_bag) > self.exp_replay_size:
            self.small_bag.popleft()

        if transition.is_terminal:
            return_t = 0
            for state, reward, mcts_pi in reversed(self.small_bag):

                return_t = reward + self.gamma * return_t
                self.big_bag.append((state, mcts_pi, return_t))

                if len(self.big_bag) > self.exp_replay_size:
                    self.big_bag.popleft()

            self.small_bag.clear()

    def store(self):
        path = self.save_data_path
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            log.warn(
                "Examples store directory does not exist! Creating directory {}".format(folder))
            os.makedirs(folder)

        with open(path, "wb+") as f:
            Pickler(f).dump(self.big_bag)

    def load(self):
        path = self.save_data_path
        if not os.path.isfile(path):
            log.warning("File with train examples was not found.")
        else:
            log.info("File with train examples found. Reading it.")
            with open(path, "rb") as f:
                self.big_bag = Unpickler(f).load()

            # Prune dataset if too big
            while len(self.big_bag) > self.exp_replay_size:
                self.big_bag.popleft()

    @property
    def metrics(self):
        logs = {"# samples": len(self.big_bag)}
        return logs

    def _create_small_package(self, transition):
        return (transition.state, transition.reward, self._recent_action_probs)


class Scoreboard(Callback):
    """Calculates agent average return from one loop (many episodes) run."""

    def on_loop_start(self):
        self.reset()

    def on_episode_start(self, episode, train_mode):
        self.episodes += 1

    def on_step_taken(self, step, transition, info):
        self.rewards += transition.reward

    def reset(self):
        self.episodes, self.rewards = 0, 0

    @property
    def metrics(self):
        return {"avg. return": self.rewards / self.episodes}
