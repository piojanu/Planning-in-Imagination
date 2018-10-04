import logging as log
import numpy as np
import os
import sys

from collections import deque
from third_party.humblerl import Callback
from pickle import Pickler, Unpickler


class Storage(Callback):
    def __init__(self, game, params={}):
        """
        Storage with train examples.

        Args:
            game (Game): Board game object.
            params (JSON Dictionary):
                * 'exp_replay_size' (int)   : Max size of big bag. When big bag is full then oldest
                                              element is removed. (Default: 100000)
                * 'save_data_path' (string) : Path to file where to store/load from big bag.
                                              (Default: "./checkpoints/data.examples")
        """

        self.game = game
        self.params = params
        self.small_bag = deque()
        self.big_bag = deque()

        self._recent_action_probs = None

    def on_action_planned(self, step, logits, info):
        # Proportional without temperature
        self._recent_action_probs = logits / np.sum(logits)

    def on_step_taken(self, step, transition, info):
        # NOTE: We never pass terminal state (it would be next_state), so NN can't learn directly
        #       what is the value of terminal/end state.
        small_package = transition.state, self._recent_action_probs
        self.small_bag.append(small_package)
        if len(self.small_bag) > self.params["exp_replay_size"]:
            self.small_bag.popleft()

        if transition.is_terminal:
            for (board, player), mcts_pi in self.small_bag:
                cannonical_state = self.game.getCanonicalForm(board, player)
                # Reward from env is from player one perspective, so we multiply reward by player
                # id which is 1 for player one or -1 player two.
                player_reward = transition.reward * player
                self.big_bag.append((cannonical_state, mcts_pi, player_reward))
                if len(self.big_bag) > self.params["exp_replay_size"]:
                    self.big_bag.popleft()
            self.small_bag.clear()

    def store(self):
        path = self.params["save_data_path"]
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            log.warn(
                "Examples store directory does not exist! Creating directory {}".format(folder))
            os.makedirs(folder)

        with open(path, "wb+") as f:
            Pickler(f).dump(self.big_bag)

    def load(self):
        path = self.params["save_data_path"]
        if not os.path.isfile(path):
            log.warning("File with train examples was not found.")
        else:
            log.info("File with train examples found. Reading it.")
            with open(path, "rb") as f:
                self.big_bag = Unpickler(f).load()

            # Prune dataset if too big
            while len(self.big_bag) > self.params["exp_replay_size"]:
                self.big_bag.popleft()

    @property
    def metrics(self):
        logs = {"# samples": len(self.big_bag)}
        return logs


class Tournament(Callback):
    def __init__(self):
        self.reset()

    def on_loop_start(self):
        self.reset()

    def on_step_taken(self, step, transition, info):
        if transition.is_terminal:
            # NOTE: Because players have fixed player id, and reward is returned from
            # perspective of player one, we are indifferent to who is starting the game.
            if transition.reward == 0:
                self.draws += 1
            elif transition.reward > 0:
                self.wins += 1
            else:
                self.losses += 1

    def reset(self):
        self.wins, self.losses, self.draws = 0, 0, 0

    @property
    def metrics(self):
        return {"wannabe": self.wins, "best": self.losses, "draws": self.draws}

    @property
    def results(self):
        return self.wins, self.losses, self.draws


class RenderCallback(Callback):
    def __init__(self, env, render, fancy=False):
        self.env = env
        self.render = render
        self.fancy = fancy

    def on_step_taken(self, step, transition, info):
        self.do_render()

    def do_render(self):
        if self.render:
            self.env.render(self.fancy)
