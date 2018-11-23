#!/usr/bin/env python3
import logging as log
import os.path

import click
import humblerl as hrl
from humblerl.agents import ChainVision, RandomAgent

from common_utils import TqdmStream, create_directory, obtain_config, mute_tf_logs_if_needed
from memory import build_rnn_model, EPNDataset, EPNVision
from utils import Config, ExperienceStorage
from world_models.vision import BasicVision, build_vae_model


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
@click.argument('path', type=click.Path(), required=True)
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-e', '--epn-path', default=None,
              help='Path to EPN-RNN ckpt. Taken from .json config if `None` (Default: None)')
def iter_train(ctx, path, vae_path, epn_path):
    """Iteratively train Expert Policy Network.

    Args:
        ctx (click.core.Context): context object.
    """

    from world_models.third_party.torchtrainer import EarlyStopping, LambdaCallback, ModelCheckpoint, CSVLogger, RandomBatchSampler
    from torch.utils.data import DataLoader

    # We will spawn multiple workers, we don't want them to access GPU
    config = obtain_config(ctx, use_gpu=False)

    # Create checkpoint/logs directory, if it doesn't exist
    create_directory(os.path.dirname(config.rnn['ckpt_path']))
    create_directory(config.rnn['logs_dir'])

    # Create env, vision and memory, mind, storage...
    env = hrl.create_gym(config.general['game_name'])
    _, encoder, _ = build_vae_model(
        config.vae, config.general['state_shape'], model_path=vae_path)
    rnn = build_rnn_model(
        config.rnn, config.vae['latent_space_dim'], env.action_space, model_path=epn_path)
    mind = RandomAgent(env)
    storage_callback = ExperienceStorage(path, config.rnn['exp_replay_size'], config.rnn['gamma'])
    storage_callback.load()

    # Create training DataLoader
    dataset = EPNDataset(storage_callback, config.rnn['sequence_len'], config.rnn['terminal_prob'])
    data_loader = DataLoader(
        dataset,
        batch_size=config.rnn['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    # Create vision
    basic_vision = BasicVision(  # Resizes states to `state_shape` with cropping
        state_shape=config.general['state_shape'],
        crop_range=config.general['crop_range']
    )
    epn_vision = EPNVision(
        vae_model=encoder,
        epn_model=rnn.model
    )
    vision = ChainVision(basic_vision, epn_vision)

    # Create training callbacks
    train_callbacks = [
        EarlyStopping(metric='loss', patience=config.rnn['patience'], verbose=1),
        LambdaCallback(on_batch_begin=lambda _, batch_size: rnn.model.init_hidden(batch_size)),
        ModelCheckpoint(config.rnn['ckpt_path'], metric='loss', save_best=True),
        CSVLogger(filename=os.path.join(config.rnn['logs_dir'], 'train_mem.csv'))
    ]

    # Train EPN for inf epochs
    while True:
        # Gather data
        hrl.loop(env, mind, vision, n_episodes=config.ctrl['n_episodes'],
                 name="Gather data", verbose=1,
                 callbacks=[epn_vision, storage_callback])
        storage_callback.store()

        # Fit EPN-RNN model!
        rnn.fit_loader(
            data_loader,
            epochs=config.rnn['epochs'],
            callbacks=train_callbacks
        )


if __name__ == '__main__':
    cli()
