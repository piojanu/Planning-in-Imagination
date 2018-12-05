from abc import ABCMeta, abstractclassmethod
import os.path

import humblerl as hrl
from humblerl.agents import ChainVision, RandomAgent
import numpy as np
from torch.utils.data import DataLoader

from common_utils import ReturnTracker, get_model_path_if_exists
from memory import build_rnn_model, EPNDataset, EPNVision
from utils import create_checkpoint_path, get_last_checkpoint_path, read_checkpoint_metadata
from utils import ExperienceStorage
from world_models.utils import MemoryVisualization
from world_models.third_party.torchtrainer import EarlyStopping, LambdaCallback
from world_models.third_party.torchtrainer import Callback, CSVLogger, TensorBoardLogger
from world_models.vision import BasicVision, build_vae_model


class Coach(Callback, metaclass=ABCMeta):
    """Coach interface, all the operations to train and run algorithm.

    Args:
        config (Config): Configuration loaded from .json file.
        vae_path (string): Path to VAE ckpt. Taken from .json config if `None` (Default: None)
        epn_path (string): Path to EPN ckpt. Taken from .json config if `None` (Default: None)
        train_mode (bool): If Coach is in train mode (can be trained) or in eval mode (can only
            be played).

    Attributes:
        config (Config): Configuration loaded from .json file.
        env (hrl.Environment): Environment to play in.
        mind (hrl.Mind): Agent's mind.
        vision (hrl.Vision): state and reward preprocessing.
        data_loader (torch.utils.data.DataLoader) Gathered data in torch loader.
        trainer (TorchTrainer): Expert Prediction Network trainer.
        storage (Storage): Experience replay buffer.
        play_callbacks (list): Play phase callbacks for `hrl.loop(...)`.
        train_callbacks (list): Train phase callbacks for `TorchTrainer.fit(...)`.
        global_epoch (int): Current epoch of training (play->train->evaluate iteration).
        best_score (float): Current best agent score.
        train_mode (bool): If Coach is in train or eval mode.

    Note:
        Do not override play and train callbacks lists! Append new callbacks to them.
    """

    def __init__(self, config, vae_path=None, epn_path=None, train_mode=True):

        self.config = config
        self.train_mode = train_mode
        self.mind = None

        # Get ckpt path and metadata
        default_path = get_last_checkpoint_path(self.config.rnn['ckpt_dir'])
        path = get_model_path_if_exists(epn_path, default_path or '', 'EPN-RNN')

        # Initialize metadata
        if not path:  # When path is empty
            self.iteration = 0
            self.global_epoch = 0
            self.best_score = float('-inf')
        else:
            self.iteration, self.global_epoch, self.best_score = read_checkpoint_metadata(path)

        # Track agent return, used to calculate mean in `play`
        self.play_callbacks = [ReturnTracker()]

        # Create env and mind
        self.env = hrl.create_gym(self.config.general['game_name'])

        # Create vision and memory modules
        _, encoder, decoder = build_vae_model(
            self.config.vae, self.config.general['state_shape'], model_path=vae_path)
        self.trainer = build_rnn_model(
            self.config.rnn, self.config.vae['latent_space_dim'], self.env.action_space,
            model_path=path or None)  # None when path is empty, start tabula rasa

        # Create HRL vision
        basic_vision = BasicVision(  # Resizes states to `state_shape` with cropping
            state_shape=self.config.general['state_shape'],
            crop_range=self.config.general['crop_range']
        )
        epn_vision = EPNVision(
            vae_model=encoder,
            epn_model=self.trainer.model
        )

        self.vision = ChainVision(basic_vision, epn_vision)
        self.play_callbacks.append(epn_vision)

        if self.train_mode:
            # Create storage and load data
            self.storage = ExperienceStorage(
                self.config.ctrl['save_data_path'],
                self.config.ctrl['exp_replay_size'],
                self.config.rnn['gamma'])
            self.storage.load()
            self.play_callbacks.append(self.storage)

            # Create training DataLoader
            dataset = EPNDataset(
                self.storage, self.config.rnn['sequence_len'], self.config.rnn['terminal_prob'])
            self.data_loader = DataLoader(
                dataset,
                batch_size=self.config.rnn['batch_size'],
                shuffle=True,
                pin_memory=True
            )

            # Create training callbacks
            self.train_callbacks = [
                self,  # Include self to count epochs
                EarlyStopping(metric='loss', patience=self.config.rnn['patience'], verbose=1),
                LambdaCallback(on_batch_begin=lambda _, batch_size:
                               self.trainer.model.init_hidden(batch_size)),
                CSVLogger(os.path.join(self.config.rnn['logs_dir'], 'train_mem.csv')),
                TensorBoardLogger(os.path.join(self.config.rnn['logs_dir'], 'tensorboard'))
            ]

            # Crate memory visualization callback if render allowed
            if self.config.allow_render:
                self.train_callbacks += [
                    MemoryVisualization(self.config, decoder, self.trainer.model, dataset, 'epn_plots')]
        else:  # If in eval mode
            self.storage = None
            self.data_loader = None
            self.train_callbacks = None

    def play(self, desc="Play", n_episodes=None, callbacks=None):
        """Self-play phase, gather data using best nn and save to storage.

        Args:
            desc (str): Progress bar description.
            n_episodes (int): How many games to play. If None, then taken from .json config
                (Default: None)
            callbacks (list): Play phase callbacks for `hrl.loop(...)`. (Default: None)

        Returns:
            float: Mean score after `n_episodes` from .json config.
        """

        callbacks = callbacks if callbacks else []

        hist = hrl.loop(self.env, self.mind, self.vision,
                        train_mode=self.train_mode,
                        debug_mode=self.config.is_debug,
                        render_mode=self.config.allow_render and not self.train_mode,
                        n_episodes=n_episodes if n_episodes else self.config.ctrl['n_episodes'],
                        callbacks=self.play_callbacks + callbacks,
                        name=desc, verbose=1 if self.train_mode else 2)

        if self.train_mode:
            # Store gathered data
            self.storage.store()

        return np.mean(hist['return'])

    def train(self, callbacks=None):
        """Training phase, improve neural net.

        Args:
            callbacks (list): Train phase callbacks for `TorchTrainer.fit(...)`.
        """

        assert self.train_mode, "Coach is in eval mode!"

        callbacks = callbacks if callbacks else []
        epochs = self.config.rnn['epochs'] + self.global_epoch

        self.trainer.fit_loader(
            self.data_loader,
            epochs=epochs,
            initial_epoch=self.global_epoch,
            callbacks=self.train_callbacks + callbacks,
        )

        self.iteration += 1

    def update_best(self, current_score):
        """Update best score and save ckpt if current agent is better.

        Args:
            current_score (float): Current agent's score.

        Returns:
            bool: True if current agent's score is higher, False otherwise.
        """

        if current_score > self.best_score:
            self.best_score = current_score
            path = create_checkpoint_path(self.config.rnn['ckpt_dir'],
                                          self.iteration,
                                          self.global_epoch,
                                          current_score)
            self.trainer.save_ckpt(path)
            return True
        return False

    def on_epoch_end(self, epoch, _):
        self.global_epoch = epoch + 1


class RandomCoach(Coach):
    """Random agent's coach, keeps all the operations to train and run algorithm.

    Args:
        config (Config): Configuration loaded from .json file.
        vae_path (string): Path to VAE ckpt. Taken from .json config if `None` (Default: None)
        epn_path (string): Path to EPN ckpt. Taken from .json config if `None` (Default: None)
        train_mode (bool): If Coach is in train mode (can be trained) or in eval mode (can be
            only played).
    """

    def __init__(self, config, vae_path=None, epn_path=None, train_mode=True):
        super(RandomCoach, self).__init__(config, vae_path, epn_path, train_mode)
        self.mind = RandomAgent(self.env)
