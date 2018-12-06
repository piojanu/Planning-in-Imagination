#!/usr/bin/env python3
import logging as log
import os.path

import click

from coach import AlphaZeroCoach as Coach
from common_utils import TensorBoardLogger, TqdmStream, create_directory, obtain_config
from common_utils import mute_tf_logs_if_needed
from utils import Config


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
    create_directory(os.path.dirname(config.rnn['ckpt_dir']))
    create_directory(config.rnn['logs_dir'])

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(os.path.join(config.rnn['logs_dir'], 'tensorboard'))

    # Train EPN for inf epochs
    while config.ctrl['iterations'] == -1 or coach.iteration < config.ctrl['iterations']:
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
        if len(coach.storage.big_bag) < config.ctrl["min_examples"]:
            log.warning(
                "Skip training, gather minimum %d training examples!",
                config.ctrl["min_examples"]
            )
            continue

        # Fit EPN-RNN model!
        coach.train()


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
