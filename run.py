#!/usr/bin/env python3
import logging as log
import math
import os.path

import click
import h5py
import humblerl as hrl
import numpy as np
from tqdm import tqdm

from coach import AlphaZeroCoach as Coach
from common_utils import TensorBoardLogger, TqdmStream, create_directory, obtain_config
from common_utils import mute_tf_logs_if_needed
from memory import WorldState
from utils import Config, ExperienceStorage
from world_models.vision import build_vae_model


@click.group()
@click.pass_context
@click.option('-c', '--config-path', type=click.Path(exists=False), default="config.json",
              help="Path to configuration file (Default: config.json)")
@click.option('--debug/--no-debug', default=False, help="Enable debug logging (Default: False)")
@click.option('--quiet/--no-quiet', default=False, help="Disable info logging (Default: False)")
@click.option('--render/--no-render', default=False, help="Allow to render/plot (Default: False)")
def cli(ctx, config_path, debug, quiet, render):
    # Get and set up logger level and formatter
    if quiet:
        level = log.ERROR
    elif debug:
        level = log.DEBUG
    else:
        level = log.INFO

    mute_tf_logs_if_needed()
    log.basicConfig(level=level, format="[%(levelname)s]: %(message)s", stream=TqdmStream)

    # Load configuration from .json file into ctx object
    ctx.obj = Config(config_path, debug, render)


@cli.command()
@click.pass_context
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-e', '--epn-path', default=None,
              help='Path to EPN-RNN ckpt. Taken from .json config if `None` (Default: None)')
def iter_train(ctx, vae_path, epn_path):
    """Iteratively train Expert Policy Network.

    Args:
        ctx (click.core.Context): context object.
        vae_path (string): Path to VAE ckpt. Taken from .json config if `None`. (Default: None)
        epn_path (string): Path to EPN-RNN ckpt. Taken from .json config if `None`. (Default: None)
    """

    config = obtain_config(ctx)
    coach = Coach(config, vae_path, epn_path, train_mode=True)

    # Create checkpoint/logs directory, if it doesn't exist
    create_directory(os.path.dirname(config.rnn['ckpt_path']))
    create_directory(config.rnn['logs_dir'])

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(os.path.join(config.rnn['logs_dir'], 'tensorboard'))

    # Train EPN for inf epochs
    while config.az['max_iter'] == -1 or coach.iteration < config.az['max_iter']:
        # Gather data
        mean_score = coach.play("Iter. {}".format(coach.iteration + 1))

        # Update best if current is better (and save ckpt)
        is_better = coach.update_best(mean_score)
        if is_better:
            log.info("Did save new checkpoint, current best: %f", mean_score)
        else:
            log.info("Didn't save new checkpoint, last best: %f", coach.best_score)

        # Log agent's current mean tb
        tb_logger.log_scalar("Mean score", mean_score, coach.iteration)

        # Proceed to training only if threshold is fulfilled
        if len(coach.storage.big_bag) < config.az["min_examples"]:
            log.warning(
                "Skip training, gather minimum %d training examples!",
                config.az["min_examples"]
            )
            continue

        # Fit EPN-RNN model!
        coach.train()


@cli.command()
@click.pass_context
@click.argument('path_in', type=click.Path(), required=True)
@click.argument('path_out', type=click.Path(), required=True)
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
def convert_data(ctx, path_in, path_out, vae_path):
    """Take transitions from record_data, preprocess states for Memory training using the trained
    VAE model. Data is loaded from `PATH_IN` (HDF5) and saved to `PATH_OUT` (Pickle)."""

    config = obtain_config(ctx)

    # Get number of actions
    env = hrl.create_gym(config.general['game_name'])
    actions_num = env.action_space.num
    del env

    # Create VAE encoder
    _, encoder, _ = build_vae_model(config.vae, config.general['state_shape'], vae_path)

    with h5py.File(path_in, 'r') as hdf_in:
        n_transitions = hdf_in.attrs['N_TRANSITIONS']
        chunk_size = hdf_in.attrs['CHUNK_SIZE']
        storage_out = ExperienceStorage(path_out, n_transitions, config.planner['gamma'])

        # Preprocess states from input dataset by using VAE
        log.info("Preprocessing states with VAE...")
        n_chunks = math.ceil(n_transitions / chunk_size)
        game_i = 1
        for chunk_i in tqdm(range(n_chunks), ascii=True):
            beg, end = chunk_i * chunk_size, min((chunk_i + 1) * chunk_size, n_transitions)
            batch_size = end - beg

            states_batch = hdf_in['states'][beg:end]
            # NOTE: [0] <- gets latent space mean (mu)
            enc_states = encoder.predict(states_batch / 255.)[0]

            # Convert actions to one hot vectors
            actions = hdf_in['actions'][beg:end].reshape(-1)
            actions_one_hot = np.zeros((batch_size, actions_num))
            actions_one_hot[np.arange(batch_size), actions] = 1

            rewards = hdf_in['rewards'][beg:end]

            # Put encoded states, converted actions and untouched rewards into experience storage
            for transition_i, state, action, one_hot, reward in \
                    zip(range(beg, end), enc_states, actions, actions_one_hot, rewards):
                storage_out.on_action_planned(None, one_hot, None)

                if transition_i < hdf_in['episodes'][game_i] - 1:
                    is_done = False
                else:
                    is_done = True
                    game_i += 1

                world_state = WorldState(state, None, None)
                storage_out.on_step_taken(
                    None, hrl.Transition(world_state, action, reward, world_state, is_done), None)

        storage_out.store()


@cli.command()
@click.pass_context
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-e', '--epn-path', default=None,
              help='Path to EPN-RNN ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-n', '--n-games', default=3, help='Number of games to play (Default: 3)')
def eval(ctx, vae_path, epn_path, n_games):
    """Iteratively train Expert Policy Network.

    Args:
        ctx (click.core.Context): context object.
        vae_path (string): Path to VAE ckpt. Taken from .json config if `None`. (Default: None)
        epn_path (string): Path to EPN-RNN ckpt. Taken from .json config if `None`. (Default: None)
        n_games (int): How many games to play. (Default: 3)
    """

    config = obtain_config(ctx)
    coach = Coach(config, vae_path, epn_path, train_mode=False)

    avg_return = coach.play(desc="Evaluate", n_episodes=n_games)

    print("Avg. return:", avg_return)


if __name__ == '__main__':
    cli()
