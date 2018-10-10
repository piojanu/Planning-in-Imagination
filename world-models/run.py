#!/usr/bin/env python3
import datetime as dt
import logging as log
import os
import click
import h5py as h5
import numpy as np
import third_party.humblerl as hrl

from functools import partial
from third_party.humblerl.agents import RandomAgent
from third_party.humblerl.callbacks import StoreTransitions2Hdf5
from tqdm import tqdm
from controller import build_es_model, Evaluator, ReturnTracker
from memory import build_rnn_model, MDNDataset, MDNVision, StoreTrajectories2npz
from utils import Config, HDF5DataGenerator, TqdmStream, state_processor, create_directory
from vision import build_vae_model, VAEVision


@click.group()
@click.pass_context
@click.option('-c', '--config_path', type=click.Path(exists=True), default="config.json",
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

    log.basicConfig(level=level, format="[%(levelname)s]: %(message)s", stream=TqdmStream)

    # Load configuration from .json file into ctx object
    ctx.obj = Config(config_path, debug, render)


@cli.command()
@click.pass_context
@click.argument('path', type=click.Path(), required=True)
@click.option('-n', '--n_games', default=10000, help='Number of games to play (Default: 10000)')
@click.option('-c', '--chunk_size', default=128, help='HDF5 chunk size (Default: 128)')
@click.option('-t', '--state_dtype', default='u1', help='Numpy data type of state (Default: uint8)')
def record_vae(ctx, path, n_games, chunk_size, state_dtype):
    """Plays chosen game randomly and records transitions to hdf5 file in `PATH`."""

    config = ctx.obj

    # Create Gym environment, random agent and store to hdf5 callback
    env = hrl.create_gym(config.general['game_name'])
    mind = RandomAgent(env.action_space)
    store_callback = StoreTransitions2Hdf5(
        env.valid_actions, config.general['state_shape'], path, chunk_size=chunk_size, dtype=state_dtype)

    # Resizes states to `state_shape` with cropping
    vision = hrl.Vision(partial(
        state_processor,
        state_shape=config.general['state_shape'],
        crop_range=config.general['crop_range']))

    # Play `N` random games and gather data as it goes
    hrl.loop(env, mind, vision, n_episodes=n_games, verbose=1, callbacks=[store_callback])


@cli.command()
@click.pass_context
@click.argument('path', type=click.Path(exists=True), required=True)
def train_vae(ctx, path):
    """Train VAE model as specified in .json config with data at `PATH`."""

    from keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
    config = ctx.obj

    # Get dataset length and eight examples to evaluate VAE on
    with h5.File(path, 'r') as hfile:
        n_transitions = hfile.attrs['N_TRANSITIONS']
        X_eval = hfile['states'][:8] / 255.

    # Get training data
    train_gen = HDF5DataGenerator(path, 'states', 'states', batch_size=config.vae['batch_size'],
                                  end=int(n_transitions * 0.8),
                                  preprocess_fn=lambda X, y: (X / 255., y / 255.))
    val_gen = HDF5DataGenerator(path, 'states', 'states', batch_size=config.vae['batch_size'],
                                start=int(n_transitions * 0.8),
                                preprocess_fn=lambda X, y: (X / 255., y / 255.))

    # Build VAE model
    vae, _, _ = build_vae_model(config.vae, config.general['state_shape'])

    # If render features enabled...
    if config.allow_render:
        # ...plot first eight training examples with VAE reconstructions
        # at the beginning of every epoch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt

        # Check if destination dir exists
        plots_dir = os.path.join(config.vae['logs_dir'], "plots_vae")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Evaluate VAE at the end of epoch
        def plot_samples(epoch, logs):
            pred = vae.predict(X_eval)

            samples = np.empty_like(np.concatenate((X_eval, pred)))
            samples[0::2] = X_eval
            samples[1::2] = pred

            _ = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(*config.general['state_shape']))

            # Save figure to logs dir
            plt.savefig(os.path.join(
                plots_dir,
                "vision_sample_{}".format(dt.datetime.now().strftime("%d-%mT%H:%M"))
            ))
            plt.close()
    else:
        def plot_samples(epoch, logs):
            pass

    # Create checkpoint directory, if it doesn't exist
    create_directory(os.path.dirname(config.vae['ckpt_path']))

    # Initialize callbacks
    callbacks = [
        EarlyStopping(patience=config.vae['patience']),
        LambdaCallback(on_epoch_begin=plot_samples),
        ModelCheckpoint(config.vae['ckpt_path'], verbose=1,
                        save_best_only=True, save_weights_only=True)
    ]

    # Fit VAE model!
    vae.fit_generator(
        generator=train_gen,
        validation_data=val_gen,
        epochs=config.vae['epochs'],
        use_multiprocessing=True,
        # TODO:  Make generator multi-thread.
        # NOTE:  There is no need for more then one workers, we are disk IO bound (I suppose ...)
        # NOTE2: h5py from conda should be threadsafe... but it apparently isn't and raises
        #        `OSError: Can't read data (wrong B-tree signature)` sporadically if `workers` = 1
        #        and always if `workers` > 1. That's why this generator needs to run in main thread
        #        (`workers` = 0).
        workers=0,
        max_queue_size=100,
        callbacks=callbacks
    )


@cli.command()
@click.pass_context
@click.argument('path', type=click.Path(), required=True)
@click.option('-m', '--model_path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-n', '--n_games', default=10000, help='Number of games to play (Default: 10000)')
def record_mem(ctx, path, model_path, n_games):
    """Plays chosen game randomly and records preprocessed with VAE (loaded from `--model_path`
    or config) states, next_states and actions trajectories to numpy archive file in `PATH`."""

    config = ctx.obj

    # Create Gym environment, random agent and store to hdf5 callback
    env = hrl.create_gym(config.general['game_name'])
    mind = RandomAgent(env.action_space)
    store_callback = StoreTrajectories2npz(path)

    # Build VAE model
    vae, encoder, _ = build_vae_model(config.vae, config.general['state_shape'], model_path)

    # Resizes states to `state_shape` with cropping and encode to latent space
    vision = VAEVision(encoder, state_processor_fn=partial(
        state_processor,
        state_shape=config.general['state_shape'],
        crop_range=config.general['crop_range']))

    # Play `N` random games and gather data as it goes
    hrl.loop(env, mind, vision, n_episodes=n_games, verbose=1, callbacks=[store_callback])


@cli.command()
@click.pass_context
@click.argument('path', type=click.Path(exists=True), required=True)
@click.option('-v', '--vae_path', default='DEFAULT',
              help='Path to VAE ckpt. Needed for visualization only when render is enabled.')
def train_mem(ctx, path, vae_path):
    """Train MDN-RNN model as specified in .json config with data at `PATH`."""

    from third_party.torchtrainer import EarlyStopping, LambdaCallback, ModelCheckpoint
    from torch.utils.data import DataLoader
    config = ctx.obj

    # Load data
    data_npz = np.load(path)

    states = data_npz["states"].astype(np.float32)
    actions = data_npz["actions"].astype(np.int)
    lengths = data_npz["lengths"].astype(np.int)

    env = hrl.create_gym(config.general['game_name'])

    # Create training DataLoader
    data_loader = DataLoader(
        MDNDataset(states, actions, lengths, config.rnn['sequence_len']),
        batch_size=config.rnn['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    # Build model
    rnn = build_rnn_model(config.rnn, states.shape[3], env.action_space)

    # If render features enabled...
    if config.allow_render:
        if vae_path == 'DEFAULT':
            raise ValueError("To render provide valid path to VAE checkpoint!")

        # ...plot first eight training examples with VAE reconstructions
        # at the beginning of every epoch
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import torch

        # Check if destination dir exists
        plots_dir = os.path.join(config.vae['logs_dir'], "plots_mdn")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Build VAE model and load checkpoint
        _, _, decoder = build_vae_model(config.vae,
                                        config.general['state_shape'],
                                        vae_path)
        # Prepare data
        S_eval = states[0, :400, 0]
        S_next = states[0, 1:401, 0]
        A_eval = actions[0, :400]

        # Evaluate MDN-RNN at the end of epoch
        def plot_samples(epoch):
            rnn.model.init_hidden(1)
            mu, _, pi = rnn.model(torch.from_numpy(S_eval).view(1, 400, -1),
                                  torch.from_numpy(A_eval).view(1, 400, -1))

            orig_mu = S_eval[::100]
            mean_mu = torch.sum(
                mu * pi, dim=2).detach().numpy().reshape(400, -1)[::100]
            max_mu = torch.gather(mu, dim=2, index=torch.argmax(
                pi, dim=2, keepdim=True)).detach().numpy().reshape(400, -1)[::100]
            next_mu = S_next[::100]

            orig_img = decoder.predict(orig_mu)
            mean_img = decoder.predict(mean_mu)
            max_img = decoder.predict(max_mu)
            next_img = decoder.predict(next_mu)

            samples = np.empty_like(np.concatenate((orig_img, mean_img, max_img, next_img)))
            samples[0::4] = orig_img
            samples[1::4] = mean_img
            samples[2::4] = max_img
            samples[3::4] = next_img

            _ = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(*config.general['state_shape']))

            # Save figure to logs dir
            plt.savefig(os.path.join(
                plots_dir,
                "memory_sample_{}".format(dt.datetime.now().strftime("%d-%mT%H:%M"))
            ))
            plt.close()
    else:
        def plot_samples(epoch):
            pass

    # Create checkpoint directory, if it doesn't exist
    create_directory(os.path.dirname(config.rnn['ckpt_path']))

    # Initialize callbacks
    callbacks = [
        EarlyStopping(metric='loss', patience=config.rnn['patience'], verbose=1),
        LambdaCallback(on_batch_begin=lambda _, batch_size: rnn.model.init_hidden(batch_size),
                       on_epoch_begin=plot_samples),
        ModelCheckpoint(config.rnn['ckpt_path'], metric='loss', save_best=True)
    ]

    # Fit MDN-RNN model!
    rnn.fit_loader(
        data_loader,
        epochs=config.rnn['epochs'],
        callbacks=callbacks
    )


@cli.command()
@click.pass_context
@click.option('-v', '--vae_path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-m', '--mdn_path', default=None,
              help='Path to MDN-RNN ckpt. Taken from .json config if `None` (Default: None)')
def train_ctrl(ctx, vae_path, mdn_path):
    """Plays chosen game and trains Controller on preprocessed states with VAE and MDN-RNN
    (loaded from `--vae_path` or `--mdn_path`)."""

    config = ctx.obj

    # Book keeping variables
    best_return = float('-inf')

    # Gen number of workers to run
    processes = config.es['processes']
    processes = processes if processes > 0 else None

    # Get action space size
    env = hrl.create_gym(config.general['game_name'])
    action_size = env.action_space
    del env

    # Build CMA-ES solver and linear model
    solver, _ = build_es_model(config.es,
                               config.vae['latent_space_dim'] + config.rnn['hidden_units'],
                               action_size)

    # Train for N epochs
    pbar = tqdm(range(config.es['epochs']), ascii=True)
    for _ in pbar:
        # Get new population
        population = solver.ask()

        # Evaluate population in parallel
        hists = hrl.pool(
            Evaluator(config,
                      config.vae['latent_space_dim'] + config.rnn['hidden_units'],
                      action_size, vae_path, mdn_path),
            jobs=population,
            processes=processes,
            n_episodes=config.es['n_episodes'],
            verbose=0
        )
        returns = [np.mean(hist['return']) for hist in hists]

        # Print logs and update best return
        pbar.set_postfix(best=best_return, current=max(returns))
        best_return = max(best_return, max(returns))

        # Update solver
        solver.tell(returns)

        if config.es['ckpt_path']:
            # Save solver in given path
            solver.save_ckpt(config.es['ckpt_path'])
            log.debug("Saved checkpoint in path: %s", config.es['ckpt_path'])


@cli.command()
@click.pass_context
@click.option('-v', '--vae_path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-m', '--mdn_path', default=None,
              help='Path to MDN-RNN ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-c', '--cma_path', default=None,
              help='Path to CMA-ES ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-n', '--n_games', default=3, help='Number of games to play (Default: 3)')
def eval(ctx, vae_path, mdn_path, cma_path, n_games):
    """Plays chosen game testing whole pipeline: VAE -> MDN-RNN -> CMA-ES
    (loaded from `vae_path`, `mdn_path` and `cma-path`)."""

    config = ctx.obj

    # Gen number of workers to run
    processes = config.es['processes']
    processes = processes if processes > 0 else None

    # Get action space size
    env = hrl.create_gym(config.general['game_name'])
    action_size = len(env.valid_actions)

    # Create VAE + MDN-RNN vision
    _, encoder, _ = build_vae_model(config.vae,
                                    config.general['state_shape'],
                                    vae_path)

    rnn = build_rnn_model(config.rnn,
                          config.vae['latent_space_dim'],
                          action_size,
                          mdn_path)

    vision = MDNVision(encoder, rnn.model, config.vae['latent_space_dim'],
                       state_processor_fn=partial(
                           state_processor,
                           state_shape=config.general['state_shape'],
                           crop_range=config.general['crop_range']))

    # Build CMA-ES solver and linear model
    _, mind = build_es_model(config.es,
                             config.vae['latent_space_dim'] + config.rnn['hidden_units'],
                             action_size,
                             cma_path)

    hist = hrl.loop(env, mind, vision,
                    n_episodes=n_games, render_mode=config.allow_render, verbose=1,
                    callbacks=[ReturnTracker(), vision])
    print("Returns:", *hist['return'])
    print("Avg. return:", np.mean(hist['return']))


if __name__ == '__main__':
    cli()
