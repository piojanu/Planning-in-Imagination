#!/usr/bin/env python3
import click
import datetime as dt
import h5py as h5
import humblerl as hrl
import logging as log
import numpy as np
import os
import tensorflow

from common_utils import limit_gpu_memory_usage, mute_tf_logs_if_needed, create_directory, force_cpu
from controller import build_es_model, build_mind, Evaluator, ReturnTracker
from functools import partial
from humblerl.agents import RandomAgent
from memory import build_rnn_model, MDNDataset, MDNVision
from third_party.torchtrainer import RandomBatchSampler, evaluate
from tqdm import tqdm
from utils import Config, HDF5DataGenerator, TqdmStream, state_processor, StoreTransitions, convert_data_with_vae
from vision import build_vae_model


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
def record_data(ctx, path, n_games, chunk_size, state_dtype):
    """Plays chosen game randomly and records transitions to hdf5 file in `PATH`."""

    config = obtain_config(ctx)

    # Create Gym environment, random agent and store to hdf5 callback
    env = hrl.create_gym(config.general['game_name'])
    mind = RandomAgent(env)
    store_callback = StoreTransitions(path, config.general['state_shape'], env.action_space,
                                      chunk_size=chunk_size, state_dtype=state_dtype)

    if store_callback.game_count >= n_games:
        log.warning("Data is already fully present in dataset you specified! If you wish to create"
                    " a new dataset, please remove the one under this path or specify a different"
                    " path. If you wish to gather more data, increase the number of games to "
                    " record with --n-games parameter.")
        return
    elif 0 < store_callback.game_count < n_games:
        diff = n_games - store_callback.game_count
        log.info("{}/{} games were already recorded in specified dataset. {} more game will be"
                 " added!".format(store_callback.game_count, n_games, diff))
        n_games = diff

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
        shuffle=True,  # It shuffles whole batches, not items in batches
        callbacks=callbacks
    )


@cli.command()
@click.pass_context
@click.argument('path_in', type=click.Path(), required=True)
@click.argument('path_out', type=click.Path(), required=True)
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
def convert_data(ctx, path_in, path_out, vae_path):
    """Use transitions from record_data and preprocess states for Memory training
    using a trained VAE model. Data is loaded from `PATH_IN` and saved to `PATH_OUT`"""

    config = obtain_config(ctx)

    # Build VAE model
    _, encoder, _ = build_vae_model(config.vae, config.general['state_shape'], vae_path)

    convert_data_with_vae(encoder, path_in, path_out, config.vae['latent_space_dim'])


@cli.command()
@click.pass_context
@click.argument('path', type=click.Path(exists=True), required=True)
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Needed for visualization only when render is enabled.')
def train_mem(ctx, path, vae_path):
    """Train MDN-RNN model as specified in .json config with data at `PATH`."""

    from third_party.torchtrainer import EarlyStopping, LambdaCallback, ModelCheckpoint
    from torch.utils.data import DataLoader
    config = obtain_config(ctx)

    env = hrl.create_gym(config.general['game_name'])

    # Create training DataLoader
    dataset = MDNDataset(path, config.rnn['sequence_len'], config.rnn['terminal_prob'])
    data_loader = DataLoader(
        dataset,
        batch_sampler=RandomBatchSampler(dataset, config.rnn['batch_size']),
        pin_memory=True
    )

    # Build model
    rnn = build_rnn_model(config.rnn, config.vae['latent_space_dim'], env.action_space)

    # Evaluate and visualize memory progress
    if config.allow_render:
        if vae_path is None:
            raise ValueError("To render provide valid path to VAE checkpoint!")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import torch

        # Check if destination dir exists
        plots_dir = os.path.join(config.rnn['logs_dir'], "plots_mdn")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        # Build VAE model and load checkpoint
        _, _, decoder = build_vae_model(config.vae,
                                        config.general['state_shape'],
                                        vae_path)
        # Prepare data
        n_episodes = min(config.rnn["rend_n_episodes"], len(dataset))
        S_eval = np.zeros((n_episodes, dataset.sequence_len, dataset.latent_dim),
                          dtype=dataset.dataset['states'].dtype)
        A_eval = np.zeros((n_episodes, dataset.sequence_len, dataset.action_dim),
                          dtype=dataset.dataset['actions'].dtype)
        for i in range(n_episodes):
            (states, actions), _ = dataset[i]
            S_eval[i] = states
            A_eval[i] = actions

        # Evaluate MDN-RNN at the end of epoch
        def plot_samples(_):
            with evaluate(rnn.model) as net:
                # Initialize memory module
                net.init_hidden(n_episodes)

                # Initialize hidden state (warm-up memory module)
                seq_half = dataset.sequence_len // 2
                with torch.no_grad():
                    net(
                        torch.from_numpy(S_eval[:, :seq_half]).to(
                            next(net.parameters()).device),
                        torch.from_numpy(A_eval[:, :seq_half]).to(
                            next(net.parameters()).device)
                    )

            orig_mu = S_eval[:, seq_half, :]
            pred_mu = rnn.model.simulate(
                torch.from_numpy(np.expand_dims(orig_mu, axis=1)).to(  # Adds sequence dim.
                    next(rnn.model.parameters()).device),
                torch.from_numpy(A_eval[:, seq_half:seq_half + config.rnn["rend_n_rollouts"]]).to(
                    next(rnn.model.parameters()).device)
            ).reshape(-1, dataset.latent_dim)

            orig_img = decoder.predict(orig_mu)[:, np.newaxis]
            pred_img = decoder.predict(pred_mu).reshape(n_episodes,
                                                        config.rnn["rend_n_rollouts"],
                                                        *config.general['state_shape'])

            samples = np.concatenate((orig_img, pred_img), axis=1)

            fig = plt.figure(figsize=(
                config.rnn["rend_n_rollouts"] + 1,
                n_episodes + 1))  # Add + 1 to make space for titles
            gs = gridspec.GridSpec(n_episodes,
                                   config.rnn["rend_n_rollouts"] + 1,
                                   wspace=0.05, hspace=0.05, figure=fig)

            for i in range(n_episodes):
                for j in range(config.rnn["rend_n_rollouts"] + 1):
                    ax = plt.subplot(gs[i, j])
                    plt.axis('off')
                    ax.set_aspect('equal')
                    if i == 0:
                        if j == 0:
                            ax.set_title("start")
                        else:
                            ax.set_title("t + {}".format(j))
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

    input_dim = config.vae['latent_space_dim'] + config.rnn['hidden_units']
    out_dim = action_space.num
    n_params = (input_dim + 1) * out_dim
    # Build CMA-ES solver
    solver = build_es_model(config.es, n_params=n_params)

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

        # Save solver in given path
        solver.save_es_ckpt_and_mind_weights(config.es['ckpt_path'], config.es['mind_path'])
        log.debug("Saved CMA-ES checkpoint in path: %s", config.es['ckpt_path'])
        log.debug("Saved Mind weights in path: %s", config.es['mind_path'])


@cli.command()
@click.pass_context
@click.option('-c', '--controller-path', required=True,
              help='Path to Mind weights.')
@click.option('-v', '--vae-path', default=None,
              help='Path to VAE ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-m', '--mdn-path', default=None,
              help='Path to MDN-RNN ckpt. Taken from .json config if `None` (Default: None)')
@click.option('-n', '--n-games', default=3, help='Number of games to play (Default: 3)')
def eval(ctx, controller_path, vae_path, mdn_path, n_games):
    """Plays chosen game testing whole pipeline: VAE -> MDN-RNN -> Controller
    (loaded from `vae_path`, `mdn_path` and `controller_path`)."""

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
    mind = build_mind(config.es,
                      config.vae['latent_space_dim'] + config.rnn['hidden_units'],
                      env.action_space,
                      controller_path)

    hist = hrl.loop(env, mind, vision,
                    n_episodes=n_games, render_mode=config.allow_render, verbose=1,
                    callbacks=[ReturnTracker(), vision])

    print("Returns:", *hist['return'])
    print("Avg. return:", np.mean(hist['return']))


if __name__ == '__main__':
    cli()
