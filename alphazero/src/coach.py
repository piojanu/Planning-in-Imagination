import click
import third_party.humblerl as hrl
import logging as log
import numpy as np
import utils

from abc import ABCMeta, abstractmethod
from keras.callbacks import EarlyStopping, TensorBoard
from tabulate import tabulate

from algos.alphazero import Planner
from algos.board_games import AdversarialMinds, BoardRender, BoardStorage, BoardVision, Tournament, ELOScoreboard
from algos.human import HumanPlayer
from nn import build_keras_nn, KerasNet
from third_party.humblerl.callbacks import BasicStats, CSVSaverWrapper
from utils import Config, TensorBoardLogger


class Coach(object):
    """AlphaZero coach, all the operations to train and evaluate algorithm.

    Attributes:
        cfg (Config): Configuration loaded from .json file.
        env (hrl.Environment): Environment to play in.
        vision (hrl.Vision): state and reward preprocessing.
        best_nn (NeuralNet): Value and Policy network of best agent.
        current_nn (NeuralNet): Value and Policy network of current agent.
        best_mind (hrl.Mind): Best agent's mind.
        current_mind (hrl.Mind): Current agent's mind.
        storage (Storage): Experience replay buffer.
        scoreboard (hrl.Callback): Callback that measure agent's score.
        play_callbacks (list): Play phase callbacks for `hrl.loop(...)`.
        train_callbacks (list): Train phase callbacks for `NeuralNet.train(...)`.
        eval_callbacks (list): Evaluation phase callbacks for `hrl.loop(...)`.
        global_epoch (int): Current epoch of training (play->train->evaluate iteration).
        best_score (float): Current best agent score.
    """

    def __init__(self):
        """Initialize AlphaZero coach object."""

        self.cfg = None
        self.env = None
        self.vision = None
        self.best_nn = None
        self.current_nn = None
        self.best_mind = None
        self.current_mind = None
        self.storage = None
        self.scoreboard = None
        self.play_callbacks = None
        self.train_callbacks = None
        self.eval_callbacks = None
        self.global_epoch = None
        self.best_score = None

    def play(self, desc="Play"):
        """Self-play phase, gather data using best nn and save to storage.

        Args:
            desc (str): Progress bar description.
        """

        hrl.loop(self.env, self.best_mind, self.vision, policy='proportional', trian_mode=True,
                 warmup=self.cfg.self_play['policy_warmup'],
                 debug_mode=self.cfg.debug, n_episodes=self.cfg.self_play['n_self_plays'],
                 name=desc, verbose=1,
                 callbacks=[self.best_mind, self.storage, *self.play_callbacks])

        # Store gathered data
        self.storage.store()

    def train(self):
        """Training phase, improve neural net."""

        trained_data = self.storage.big_bag
        boards_input, target_pis, target_values = list(zip(*trained_data))

        self.global_epoch = self.current_nn.train(data=np.array(boards_input),
                                                  targets=[np.array(target_pis),
                                                           np.array(target_values)],
                                                  initial_epoch=self.global_epoch,
                                                  callbacks=self.train_callbacks)

    def evaluate(self, desc="Evaluation"):
        """Evaluation phase, check how good current mind is.

        Args:
            desc (str): Progress bar description.

        Note:
            `self.scoreboard` should measure and keep performance of mind
            from last call to `hrl.loop`.
        """
        # Clear current agent tree and evaluate it
        self.current_mind.clear_tree()
        hrl.loop(self.env, self.current_mind, self.vision, policy='deterministic', train_mode=False,
                 debug_mode=self.cfg.debug, n_episodes=self.cfg.self_play['n_tournaments'],
                 name=desc, verbose=2,
                 callbacks=[self.scoreboard, *self.eval_callbacks])

    def update_best(self, best_score):
        """Updates best score and saves current nn as new best.

        Args:
            best_score (float): New best agent score.
        """

        self.best_score = best_score

        # Create checkpoint file name and log it
        best_fname = utils.create_checkpoint_file_name('self_play',
                                                       self.cfg.self_play["game"],
                                                       self.global_epoch,
                                                       self.best_score)
        log.info("New best player: %s", best_fname)

        # Save best and exchange weights
        self.current_nn.save_checkpoint(self.cfg.logging['save_checkpoint_folder'], best_fname)
        self.best_nn.model.set_weights(self.current_nn.model.get_weights())


class Builder(metaclass=ABCMeta):
    """Builds AlphaZero coach components."""

    def __init__(self, config):
        """Initialize AlphaZero coach.

        Args:
            config (Config): Configuration loaded from .json file.
        """

        self.cfg = config
        self.coach = None

    def direct(self):
        """Direct AlphaZero coach building process."""

        self.coach = Coach()
        self.coach.cfg = self.cfg

        self.build_env()
        self.build_vision()
        self.build_nn()
        self.build_mind()
        self.build_storage()
        self.build_scoreboard()
        self.build_callbacks()

        # Load best nn if available
        try:
            ckpt_path = utils.get_newest_ckpt_fname(
                self.cfg.logging['save_checkpoint_folder'])
            self.coach.best_nn.load_checkpoint(
                self.cfg.logging['save_checkpoint_folder'], ckpt_path)
            log.info("Best mind has loaded latest checkpoint: %s",
                     utils.get_newest_ckpt_fname(self.cfg.logging['save_checkpoint_folder']))
            self.coach.global_epoch = utils.get_checkpoints_epoch(ckpt_path)
            self.coach.best_score = utils.get_checkpoints_elo(ckpt_path)
        except:
            log.info("No initial checkpoint, starting tabula rasa.")
            self.build_metadata()

        # Copy best nn weights to current nn that will be trained
        self.coach.current_nn.model.set_weights(self.coach.best_nn.model.get_weights())

        return self.coach

    @abstractmethod
    def build_env(self):
        pass

    @abstractmethod
    def build_vision(self):
        pass

    @abstractmethod
    def build_nn(self):
        pass

    @abstractmethod
    def build_metadata(self):
        pass

    @abstractmethod
    def build_mind(self):
        pass

    @abstractmethod
    def build_storage(self):
        pass

    @abstractmethod
    def build_scoreboard(self):
        pass

    @abstractmethod
    def build_callbacks(self):
        pass


class BoardGameBuilder(Builder):
    """ALphaZero coach builder for board games."""

    def build_env(self):
        self.coach.env = self.cfg.env

    def build_vision(self):
        self.coach.vision = BoardVision(self.cfg.game)

    def build_nn(self):
        self.coach.best_nn = KerasNet(build_keras_nn(
            self.cfg.game, self.cfg.nn), self.cfg.training)
        self.coach.current_nn = KerasNet(build_keras_nn(
            self.cfg.game, self.cfg.nn), self.cfg.training)

    def build_metadata(self):
        self.coach.global_epoch = 0
        self.coach.best_score = 1000

    def build_mind(self):
        best_player = Planner(self.cfg.mdp, self.coach.best_nn, self.cfg.planner)
        current_player = Planner(self.cfg.mdp, self.coach.current_nn, self.cfg.planner)

        self.coach.best_mind = AdversarialMinds(best_player, best_player)
        self.coach.current_mind = AdversarialMinds(current_player, best_player)

    def build_storage(self):
        self.coach.storage = BoardStorage(self.cfg)
        # Load storage date from disk (path in config)
        self.coach.storage.load()

    def build_scoreboard(self):
        self.coach.scoreboard = CSVSaverWrapper(
            Tournament(), self.cfg.logging['save_tournament_log_path'], True)

    def build_callbacks(self):
        self.coach.play_callbacks = [
            CSVSaverWrapper(BasicStats(), self.cfg.logging['save_self_play_log_path'])]
        self.coach.train_callbacks = [
            TensorBoard(log_dir=utils.create_tensorboard_log_dir(
                self.cfg.logging['tensorboard_log_folder'], 'self_play'))]
        self.coach.eval_callbacks = [self.cfg.env]  # env added to alternate starting player
