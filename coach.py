from abc import ABCMeta, abstractclassmethod
import os.path

import humblerl as hrl
from humblerl.agents import ChainVision, RandomAgent
from torch.utils.data import DataLoader

from memory import build_rnn_model, EPNDataset, EPNVision
from utils import ExperienceStorage
from world_models.third_party.torchtrainer import EarlyStopping, LambdaCallback, ModelCheckpoint, CSVLogger
from world_models.vision import BasicVision, build_vae_model


class Coach(metaclass=ABCMeta):
    """Coach interface, all the operations to train and run algorithm.

    Args:
        config (Config): Configuration loaded from .json file.

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
    """

    def __init__(self, config):

        self.config = config
        self.env = None
        self.mind = None
        self.vision = None
        self.data_loader = None
        self.trainer = None
        self.storage = None
        self.play_callbacks = []
        self.train_callbacks = []
        self.global_epoch = 0
        self.best_score = 0

    def play(self, desc="Gather data", train_mode=True, callbacks=None):
        """Self-play phase, gather data using best nn and save to storage.

        Args:
            desc (str): Progress bar description.
            train_mode (bool): Informs whether this run is in training or evaluation mode.
            callbacks (list): Play phase callbacks for `hrl.loop(...)`.
        """

        callbacks = callbacks if callbacks else []

        hrl.loop(self.env, self.mind, self.vision,
                 train_mode=train_mode,
                 debug_mode=self.config.is_debug,
                 n_episodes=self.config.ctrl['n_episodes'],
                 callbacks=self.play_callbacks + callbacks,
                 name=desc, verbose=1)

        # Store gathered data
        self.storage.store()

    def train(self, callbacks=None):
        """Training phase, improve neural net.

        Args:
            callbacks (list): Train phase callbacks for `TorchTrainer.fit(...)`.
        """

        callbacks = callbacks if callbacks else []

        self.trainer.fit_loader(
            self.data_loader,
            epochs=self.config.rnn['epochs'],
            initial_epoch=self.global_epoch,
            callbacks=self.train_callbacks + callbacks,
        )


class RandomCoach(Coach):
    """Random agent coach interface.

    Args:
        config (Config): Configuration loaded from .json file.
        vae_path (string): Path to VAE ckpt. Taken from .json config if `None` (Default: None)
        epn_path (string): Path to EPN ckpt. Taken from .json config if `None` (Default: None)
    """

    def __init__(self, config, vae_path=None, epn_path=None):
        super(RandomCoach, self).__init__(config)

        # Create env and mind
        self.env = hrl.create_gym(config.general['game_name'])
        self.mind = RandomAgent(self.env)

        # Create storage and load data
        self.storage = ExperienceStorage(
            config.ctrl['save_data_path'], config.ctrl['exp_replay_size'], config.rnn['gamma'])
        self.storage.load()
        self.play_callbacks.append(self.storage)

        # Create training DataLoader
        dataset = EPNDataset(
            self.storage, config.rnn['sequence_len'], config.rnn['terminal_prob'])
        self.data_loader = DataLoader(
            dataset,
            batch_size=config.rnn['batch_size'],
            shuffle=True,
            pin_memory=True
        )

        # Create vision
        _, encoder, _ = build_vae_model(
            config.vae, config.general['state_shape'], model_path=vae_path)
        self.trainer = build_rnn_model(
            config.rnn, config.vae['latent_space_dim'], self.env.action_space, model_path=epn_path)

        basic_vision = BasicVision(  # Resizes states to `state_shape` with cropping
            state_shape=config.general['state_shape'],
            crop_range=config.general['crop_range']
        )
        epn_vision = EPNVision(
            vae_model=encoder,
            epn_model=self.trainer.model
        )

        self.vision = ChainVision(basic_vision, epn_vision)
        self.play_callbacks.append(epn_vision)

        # Create training callbacks
        self.train_callbacks = [
            EarlyStopping(metric='loss', patience=config.rnn['patience'], verbose=1),
            LambdaCallback(on_batch_begin=lambda _, batch_size:
                           self.trainer.model.init_hidden(batch_size)),
            ModelCheckpoint(config.rnn['ckpt_path'], metric='loss', save_best=True),
            CSVLogger(filename=os.path.join(config.rnn['logs_dir'], 'train_mem.csv'))
        ]
