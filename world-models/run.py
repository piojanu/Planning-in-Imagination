#!/usr/bin/env python3
import datetime as dt
import logging as log
import os
import click
import h5py as h5
import numpy as np
import humblerl as hrl

from functools import partial
from humblerl.agents import RandomAgent
from tqdm import tqdm
import tensorflow
from controller import build_es_model, Evaluator, ReturnTracker
from memory import build_rnn_model, MDNDataset, MDNVision, StoreMemTransitions
from utils import Config, HDF5DataGenerator, TqdmStream, state_processor, create_directory, force_cpu
from utils import limit_gpu_memory_usage, mute_tf_logs_if_needed
from vision import build_vae_model, VAEVision, StoreVaeTransitions


def obtain_config(ctx, use_gpu=True):
    if use_gpu:
        limit_gpu_memory_usage()
    return ctx.obj


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
@click.option('-n', '--n-games', default=10000, help='Number of games to play (Default: 10000)')
@click.option('-c', '--chunk-size', default=128, help='HDF5 chunk size (Default: 128)')
@click.option('-t', '--state-dtype', default='u1', help='Numpy data type of state (Default: uint8)')
def record_vae(ctx, path, n_games, chunk_size, state_dtype):
    """Plays chosen game randomly and records transitions to hdf5 file in `PATH`."""

    config = obtain_config(ctx)

    # Create Gym environment, random agent and store to hdf5 callback
    env = hrl.create_gym(config.general['game_name'])
    mind = RandomAgent(env)
    store_callback = StoreVaeTransitions(config.general['state_shape'], path,
                                         chunk_size=chunk_size, dtype=state_dtype)

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
    config = obtain_config(ctx)

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
@click.option('-m', '--model-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-c', '--chunk-size', default=128, help='HDF5 chunk size (Default: 128)')
@click.option('-n', '--n-games', default=10000, help='Number of games to play (Default: 10000)')
def record_mem(ctx, path, model_path, chunk_size, n_games):
    """Plays chosen game randomly and records preprocessed with VAE (loaded from `model_path`
    or config) states, next_states and actions trajectories to HDF5 file in `PATH`."""

    config = obtain_config(ctx)

    # Create Gym environment, random agent and store to hdf5 callback
    env = hrl.create_gym(config.general['game_name'])
    mind = RandomAgent(env)
    store_callback = StoreMemTransitions(path, latent_dim=config.vae['latent_space_dim'],
                                         action_space=env.action_space, chunk_size=chunk_size)

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
@click.option('-v', '--vae-path', default='DEFAULT',
              help='Path to VAE ckpt. Needed for visualization only when render is enabled.')
def train_mem(ctx, path, vae_path):
    """Train MDN-RNN model as specified in .json config with data at `PATH`."""

    from third_party.torchtrainer import EarlyStopping, LambdaCallback, ModelCheckpoint
    from torch.utils.data import DataLoader
    config = obtain_config(ctx)

    env = hrl.create_gym(config.general['game_name'])

    # Create training DataLoader
    dataset = MDNDataset(path, config.rnn['sequence_len'])
    data_loader = DataLoader(
        dataset,
        batch_size=config.rnn['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    # Build model
    rnn = build_rnn_model(config.rnn, config.vae['latent_space_dim'], env.action_space)

    # Evaluate and visualize memory progress
    if config.allow_render:
        if vae_path == 'DEFAULT':
            raise ValueError("To render provide valid path to VAE checkpoint!")

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
        S_eval = np.zeros((config.rnn["rend_n_episodes"], dataset.sequence_len, dataset.latent_dim),
                          dtype=dataset.dataset['states'].dtype)
        S_next = np.zeros_like(S_eval)
        A_eval = np.zeros((config.rnn["rend_n_episodes"], dataset.sequence_len, dataset.action_dim),
                          dtype=dataset.dataset['actions'].dtype)
        for i in range(config.rnn["rend_n_episodes"]):
            (states, actions), (next_states,) = dataset[i]
            S_eval[i] = states
            S_next[i] = next_states
            A_eval[i] = actions

        # Evaluate MDN-RNN at the end of epoch
        def plot_samples(_):
            rnn.model.init_hidden(config.rnn["rend_n_episodes"])
            mu, _, pi = rnn.model(torch.from_numpy(S_eval), torch.from_numpy(A_eval))

            seq_half = dataset.sequence_len // 2
            orig_mu = S_eval[:, seq_half]
            next_mu = S_next[:, seq_half + config.rnn["rend_n_rollouts"] - 1]
            if config.rnn["render_type"] == "mean":
                pred_mu = torch.sum(mu * pi, dim=2).detach().numpy()
            elif config.rnn["render_type"] == "max":
                pred_mu = torch.gather(mu, dim=2, index=torch.argmax(
                    pi, dim=2, keepdim=True)).detach().numpy()
            else:
                raise ValueError("Unknown render type")
            pred_mu = pred_mu[:, seq_half:seq_half + config.rnn["rend_n_rollouts"]].reshape(
                -1, dataset.latent_dim)

            orig_img = decoder.predict(orig_mu)[:, np.newaxis]
            next_img = decoder.predict(next_mu)[:, np.newaxis]
            pred_img = decoder.predict(pred_mu)
            pred_img = pred_img.reshape(config.rnn["rend_n_episodes"],
                                        config.rnn["rend_n_rollouts"],
                                        *pred_img.shape[1:])

            samples = np.concatenate((orig_img, pred_img, next_img), axis=1)

            fig = plt.figure(figsize=(
                config.rnn["rend_n_episodes"],
                config.rnn["rend_n_rollouts"] + 2))
            gs = gridspec.GridSpec(config.rnn["rend_n_episodes"],
                                   config.rnn["rend_n_rollouts"] + 2,
                                   wspace=0.05, hspace=0.05, figure=fig)

            for i in range(config.rnn["rend_n_episodes"]):
                for j in range(config.rnn["rend_n_rollouts"] + 2):
                    ax = plt.subplot(gs[i, j])
                    plt.axis('off')
                    ax.set_aspect('equal')
                    plt.imshow(samples[i, j])

            # Save figure to logs dir
            plt.savefig(os.path.join(
                plots_dir,
                "memory_sample_{}".format(dt.datetime.now().strftime("%d-%mT%H:%M:%S"))
            ))
            plt.close()
    else:
        def plot_samples(_):
            pass

    # Create checkpoint directory, if it doesn't exist
    create_directory(os.path.dirname(config.rnn['ckpt_path']))

    # Create callbacks
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

    dataset.close()


@cli.command()
@click.pass_context
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-m', '--mdn-path', default=None,
              help='Path to MDN-RNN ckpt. Taken from .json config if `None` (Default: None)')
def train_ctrl(ctx, vae_path, mdn_path):
    """Plays chosen game and trains Controller on preprocessed states with VAE and MDN-RNN
    (loaded from `vae_path` or `mdn_path`)."""

    config = obtain_config(ctx, use_gpu=False)

    # Book keeping variables
    best_return = float('-inf')

    # Gen number of workers to run
    processes = config.es['processes']
    processes = processes if processes > 0 else None

    # We will spawn multiple workers, we don't want them to access GPU
    force_cpu()

    # Get action space size
    env = hrl.create_gym(config.general['game_name'])
    action_space = env.action_space
    del env

    # Build CMA-ES solver and linear model
    solver, _ = build_es_model(config.es,
                               config.vae['latent_space_dim'] + config.rnn['hidden_units'],
                               action_space)

    # Train for N epochs
    pbar = tqdm(range(config.es['epochs']), ascii=True)
    for _ in pbar:
        # Get new population
        population = solver.ask()

        # Evaluate population in parallel
        hists = hrl.pool(
            Evaluator(config,
                      config.vae['latent_space_dim'] + config.rnn['hidden_units'],
                      action_space, vae_path, mdn_path),
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
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-m', '--mdn-path', default=None,
              help='Path to MDN-RNN ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-c', '--cma-path', default=None,
              help='Path to CMA-ES ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-n', '--n-games', default=3, help='Number of games to play (Default: 3)')
def eval(ctx, vae_path, mdn_path, cma_path, n_games):
    """Plays chosen game testing whole pipeline: VAE -> MDN-RNN -> CMA-ES
    (loaded from `vae_path`, `mdn_path` and `cma-path`)."""

    config = obtain_config(ctx)

    # Gen number of workers to run
    processes = config.es['processes']
    processes = processes if processes > 0 else None

    # Get action space size
    env = hrl.create_gym(config.general['game_name'])

    # Create VAE + MDN-RNN vision
    _, encoder, _ = build_vae_model(config.vae,
                                    config.general['state_shape'],
                                    vae_path)

    rnn = build_rnn_model(config.rnn,
                          config.vae['latent_space_dim'],
                          env.action_space,
                          mdn_path)

    vision = MDNVision(encoder, rnn.model, config.vae['latent_space_dim'],
                       state_processor_fn=partial(
                           state_processor,
                           state_shape=config.general['state_shape'],
                           crop_range=config.general['crop_range']))

    # Build CMA-ES solver and linear model
    _, mind = build_es_model(config.es,
                             config.vae['latent_space_dim'] + config.rnn['hidden_units'],
                             env.action_space,
                             cma_path)

    hist = hrl.loop(env, mind, vision,
                    n_episodes=n_games, render_mode=config.allow_render, verbose=1,
                    callbacks=[ReturnTracker(), vision])
    print("Returns:", *hist['return'])
    print("Avg. return:", np.mean(hist['return']))


if __name__ == '__main__':
    cli()
