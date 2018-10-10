#!/usr/bin/env python3
import third_party.humblerl as hrl
import logging as log
import numpy as np
import utils
from utils import Config, TensorBoardLogger
import click

from keras.callbacks import EarlyStopping, TensorBoard
from tabulate import tabulate

from algos.alphazero import Planner
from algos.board_games import AdversarialMinds, BoardRender, BoardStorage, BoardVision, Tournament, ELOScoreboard
from algos.human import HumanPlayer
from nn import build_keras_nn, KerasNet
from third_party.humblerl.callbacks import BasicStats, CSVSaverWrapper


@click.group()
@click.pass_context
@click.option('-c', '--config', type=click.File('r'),
              help="Path to configuration file (Default: config.json)", default="config.json")
@click.option('--debug/--no-debug', help="Enable debug logging (Default: False)", default=False)
def cli(ctx, config, debug):
    # Get and set up logger level and formatter
    log.basicConfig(level=log.DEBUG if debug else log.INFO,
                    format="[%(levelname)s]: %(message)s")

    # Create context
    ctx.obj = Config(config, debug)


@cli.command()
@click.pass_context
def self_play(ctx):
    """Train by self-play, retraining from self-played frames and changing best player when
    new trained player beats currently best player.

    Args:
        ctx (click.core.Context): context object.
            Parameters for training:
                * 'game' (string)                     : game name (Default: tictactoe)
                * 'max_iter' (int)                    : number of train process iterations
                                                        (Default: -1)
                * 'min_examples' (int)                : minimum number of examples to start training
                                                        nn, if -1 then no threshold. (Default: -1)
                * 'policy_warmup' (int)               : how many stochastic warmup steps should take
                                                        deterministic policy (Default: 12)
                * 'n_self_plays' (int)                : number of self played episodes
                                                        (Default: 100)
                * 'n_tournaments' (int)               : number of tournament episodes (Default: 20)
                * 'save_checkpoint_folder' (string)   : folder to save best models
                                                        (Default: "checkpoints")
                * 'save_checkpoint_filename' (string) : filename of best model (Default: "best")
                * 'save_self_play_log_path' (string)  : where to save self-play logs.
                                                        (Default: "./logs/self-play.log")
                * 'save_tournament_log_path' (string) : where to save tournament logs.
                                                        (Default: "./logs/tournament.log")
                * 'update_threshold' (float):         : required threshold to be new best player
                                                        (Default: 0.55)
    """
    cfg = ctx.obj

    # Create board games vision
    vision = BoardVision(cfg.game)

    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(utils.create_tensorboard_log_dir(
        cfg.logging['tensorboard_log_folder'], 'elo'))

    # Create Minds, current and best
    current_net = KerasNet(build_keras_nn(cfg.game, cfg.nn), cfg.training)
    best_net = KerasNet(build_keras_nn(cfg.game, cfg.nn), cfg.training)

    # Load best nn if available
    try:
        ckpt_path = utils.get_newest_ckpt_fname(
            cfg.logging['save_checkpoint_folder'])
        best_net.load_checkpoint(
            cfg.logging['save_checkpoint_folder'], ckpt_path)
        log.info("Best mind has loaded latest checkpoint: {}".format(
            utils.get_newest_ckpt_fname(cfg.logging['save_checkpoint_folder'])))
        global_epoch = utils.get_checkpoints_epoch(ckpt_path)
        best_elo = utils.get_checkpoints_elo(ckpt_path)
    except Exception:
        log.info("No initial checkpoint, starting tabula rasa.")
        global_epoch = 0
        best_elo = 1000

    # Copy best nn weights to current nn that will be trained
    current_net.model.set_weights(best_net.model.get_weights())

    # Create players
    best_player = Planner(cfg.mdp, best_net, cfg.planner)
    current_player = Planner(cfg.mdp, current_net, cfg.planner)

    self_play_players = AdversarialMinds(best_player, best_player)
    tournament_players = AdversarialMinds(current_player, best_player)

    # Create callbacks, storage and tournament
    storage = BoardStorage(cfg)
    train_stats = CSVSaverWrapper(
        BasicStats(), cfg.logging['save_self_play_log_path'])
    tournament_stats = CSVSaverWrapper(
        Tournament(), cfg.logging['save_tournament_log_path'], True)
    train_callbacks = [TensorBoard(log_dir=utils.create_tensorboard_log_dir(
        cfg.logging['tensorboard_log_folder'], 'self_play'))]

    # Load storage date from disk (path in config)
    storage.load()

    iter = global_epoch // current_net.epochs
    while cfg.self_play["max_iter"] == -1 or iter < cfg.self_play["max_iter"]:
        iter_counter_str = "{:03d}/{:03d}".format(iter + 1, cfg.self_play["max_iter"]) if cfg.self_play["max_iter"] > 0\
            else "{:03d}/inf".format(iter + 1)

        # SELF-PLAY - gather data using best nn
        hrl.loop(cfg.env, self_play_players, vision,
                 policy='proportional', warmup=cfg.self_play['policy_warmup'],
                 debug_mode=cfg.debug, n_episodes=cfg.self_play['n_self_plays'],
                 name="Self-play  " + iter_counter_str, verbose=1,
                 callbacks=[best_player, train_stats, storage])

        # Store gathered data
        storage.store()

        # Proceed to training only if threshold is fulfilled
        if len(storage.big_bag) <= cfg.self_play["min_examples"]:
            log.warn("Skip training, gather minimum {} training examples!".format(
                cfg.self_play["min_examples"]))
            continue

        # TRAINING - improve neural net
        trained_data = storage.big_bag
        boards_input, target_pis, target_values = list(zip(*trained_data))

        global_epoch = current_net.train(data=np.array(boards_input), targets=[np.array(
            target_pis), np.array(target_values)], initial_epoch=global_epoch, callbacks=train_callbacks)

        # ARENA - only the best will generate data!
        # Clear players tree
        current_player.clear_tree()
        best_player.clear_tree()

        hrl.loop(cfg.env, tournament_players, vision, policy='deterministic', train_mode=False,
                 debug_mode=cfg.debug, n_episodes=cfg.self_play['n_tournaments'],
                 name="Tournament " + iter_counter_str, verbose=2,
                 callbacks=[tournament_stats, cfg.env])  # env added to alternate starting player

        wins, losses, draws = tournament_stats.callbacks[0].results

        if wins > 0 and float(wins) / (wins + losses) > cfg.self_play["update_threshold"]:
            # Update ELO rating, use best player ELO as current player ELO
            # NOTE: We update it this way as we don't need exact ELO values, we just need to see
            #       how much if at all has current player improved.
            #       Decision based on: https://github.com/gcp/leela-zero/issues/354
            best_elo, _ = \
                ELOScoreboard.calculate_update(
                    best_elo, best_elo, wins, losses, draws)
            best_elo = int(best_elo)

            # Create checkpoint file name and log it
            best_fname = utils.create_checkpoint_file_name(
                'self_play', cfg.self_play["game"], global_epoch, best_elo)
            log.info("New best player: {}".format(best_fname))

            # Save best and exchange weights
            current_net.save_checkpoint(
                cfg.logging['save_checkpoint_folder'], best_fname)
            best_net.model.set_weights(current_net.model.get_weights())

        # Log current player ELO
        tb_logger.log_scalar("Best ELO", best_elo, iter)

        # Increment iterator
        iter += 1


@cli.command()
@click.pass_context
@click.option('-ckpt', '--checkpoint', help="Path to NN checkpoint, if None then start fresh (Default: None)", type=click.Path(), default=None)
@click.option('-save', '--save_dir', help="Dir where to save NN checkpoint, if None then don't save (Default: None)", type=click.Path(), default=None)
@click.option('--tensorboard/--no-tensorboard', help="Enable tensorboard logging (Default: False)", default=False)
def train(ctx, checkpoint, save_dir, tensorboard):
    """Train NN from passed configuration."""

    cfg = ctx.obj

    # Get TensorBoard log dir
    tensorboard_folder = utils.create_tensorboard_log_dir(
        cfg.logging['tensorboard_log_folder'], 'train')

    # Create Keras NN
    net = KerasNet(build_keras_nn(cfg.game, cfg.nn), cfg.training)

    # Load checkpoint nn if available
    global_epoch = 0
    if checkpoint:
        net.load_checkpoint(checkpoint)
        global_epoch = utils.get_checkpoints_epoch(checkpoint)
        current_elo = utils.get_checkpoints_elo(checkpoint)
        log.info("Loaded checkpoint: {}".format(checkpoint))

    # Create TensorBoard logging callback if enabled
    callbacks = []
    if tensorboard:
        callbacks.append(TensorBoard(log_dir=tensorboard_folder))

    # Create storage and load data
    storage = BoardStorage(cfg)
    storage.load()

    # Prepare training data
    trained_data = storage.big_bag
    boards_input, target_pis, target_values = list(zip(*trained_data))

    # Run training
    global_epoch = net.train(data=np.array(boards_input),
                             targets=[np.array(target_pis),
                                      np.array(target_values)],
                             initial_epoch=global_epoch,
                             callbacks=callbacks)

    # Save model checkpoint if path passed
    if save_dir:
        save_fname = utils.create_checkpoint_file_name(
            'train', cfg.self_play["game"], global_epoch, current_elo)
        net.save_checkpoint(save_dir, save_fname)


@cli.command()
@click.pass_context
@click.option('-n', '--n-steps', help="Number of optimization steps (Default: 100)", default=100)
def hopt(ctx, n_steps):
    """Hyper-parameter optimization of NN from passed configuration."""

    from skopt import gp_minimize
    from skopt.plots import plot_convergence
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    import matplotlib.pyplot as plt

    cfg = ctx.obj

    # Create storage and load data
    storage = BoardStorage(cfg)
    storage.load()

    # Prepare training data
    trained_data = storage.big_bag
    boards_input, target_pis, target_values = list(zip(*trained_data))

    data = np.array(boards_input)
    targets = [np.array(target_pis), np.array(target_values)]

    # Prepare search space
    space = []
    for k, v in cfg.nn.items():
        # Ignore loss in hyper-param tuning
        if isinstance(v, list) and k != "loss":
            if isinstance(v[0], float):
                space.append(Real(v[0], v[1], name=k))
            elif isinstance(v[0], int):
                space.append(Integer(v[0], v[1], name=k))
            else:
                space.append(Categorical(v, name=k))

    @use_named_args(space)
    def objective(**params):
        # Prepare neural net parameters
        for k, v in params.items():
            cfg.nn[k] = v

        # Build Keras neural net model
        model = build_keras_nn(cfg.game, cfg.nn)

        # Fit model
        history = model.fit(data, targets,
                            batch_size=cfg.training["batch_size"],
                            epochs=cfg.training['epochs'],
                            validation_split=0.2,
                            callbacks=[EarlyStopping(patience=7)],
                            verbose=0)

        return history.history['val_loss'][-1]

    # Perform hyper-parameter bayesian optimization
    model_gp = gp_minimize(objective, space, n_calls=n_steps, verbose=True)

    # Print results
    print("Best score: {}".format(model_gp.fun))
    print("Best parameters:")
    for i, dim in enumerate(space):
        print("\t{} = {}".format(dim.name, model_gp.x[i]))

    # Plot convergence
    _ = plot_convergence(model_gp)
    plt.savefig("hopt_convergence.png")


@cli.command()
@click.pass_context
@click.argument('first_model_path', nargs=1, type=click.Path(exists=True))
@click.argument('second_model_path', nargs=1, type=click.Path(exists=True))
@click.option('--render/--no-render', help="Enable rendering game (Default: True)", default=True)
@click.option('-n', '--n-games', help="Number of games (Default: 2)", default=2)
def clash(ctx, first_model_path, second_model_path, render, n_games):
    """Test two models. Play `n_games` between themselves.

        Args:
            first_model_path: (string): Path to player one model.
            second_model_path (string): Path to player two model.
    """
    cfg = ctx.obj

    # Create board games vision
    vision = BoardVision(cfg.game)

    # Create Minds, current and best
    first_player_net = KerasNet(build_keras_nn(cfg.game, cfg.nn), cfg.training)
    second_player_net = KerasNet(build_keras_nn(cfg.game, cfg.nn), cfg.training)

    first_player_net.load_checkpoint(first_model_path)
    second_player_net.load_checkpoint(second_model_path)

    tournament = Tournament()
    first_player = Planner(cfg.mdp, first_player_net, cfg.planner)
    second_player = Planner(cfg.mdp, second_player_net, cfg.planner)
    players = AdversarialMinds(first_player, second_player)
    hrl.loop(cfg.env, players, vision, policy='deterministic', n_episodes=n_games,
             train_mode=False, render_mode=render, debug_mode=cfg.debug,
             name="Test  models: {} vs {}".format(first_model_path.split(
                 "/")[-1], second_model_path.split("/")[-1]),
             callbacks=[tournament, cfg.env])

    log.info("{} vs {} results: {}".format(first_model_path.split("/")
                                           [-1], second_model_path.split("/")[-1], tournament.results))


@cli.command()
@click.pass_context
@click.argument('model_path', nargs=1, type=click.Path(exists=True))
@click.option('-n', '--n-games', help="Number of games (Default: 2)", default=2)
def human_play(ctx, model_path, n_games):
    """Play `n_games` with trained model.

        Args:
            model_path: (string): Path to trained model.
    """
    cfg = ctx.obj

    # Create board games vision
    vision = BoardVision(cfg.game)

    # Create Mind for NN oponnent
    first_player_net = KerasNet(build_keras_nn(cfg.game, cfg.nn), cfg.training)
    first_player_net.load_checkpoint(model_path)
    first_player = Planner(cfg.mdp, first_player_net, cfg.planner)
    human_player = HumanPlayer(cfg.mdp)
    players = AdversarialMinds(first_player, human_player)

    render_callback = BoardRender(cfg.env, render=True, fancy=True)

    tournament = Tournament()
    hrl.loop(cfg.env, players, vision, policy='deterministic', n_episodes=n_games, train_mode=False,
             name="Test models: {} vs HUMAN".format(model_path.split("/")[-1]),
             debug_mode=cfg.debug, callbacks=[tournament, render_callback, cfg.env])

    log.info("{} vs HUMAN results: {}".format(
        model_path.split("/")[-1], tournament.results))


@cli.command()
@click.pass_context
@click.option('-d', '--checkpoints_dir', type=click.Path(exists=True), default=None,
              help="Path to checkpoints. If None then take from config (Default: None)")
@click.option('-g', '--gap', help="Gap between versions of best model (Default: 2)", default=2)
@click.option('-sc', '--second_config', type=click.File('r'),
              help="Path to second configuration file", default=None)
def cross_play(ctx, checkpoints_dir, gap, second_config):
    """Validate trained models. Best networks play with each other."""
    cfg = ctx.obj
    second_cfg = Config(second_config) if second_config is not None else cfg

    # Create board games vision
    vision = BoardVision(cfg.game)

    # Set checkpoints_dir if not passed
    if checkpoints_dir is None:
        checkpoints_dir = cfg.logging['save_checkpoint_folder']

    # Create players and their minds
    first_player_net = KerasNet(build_keras_nn(cfg.game, cfg.nn), cfg.training)
    second_player_net = KerasNet(build_keras_nn(
        second_cfg.game, second_cfg.nn), second_cfg.training)
    first_player = Planner(cfg.mdp, first_player_net, cfg.planner)
    second_player = Planner(second_cfg.mdp, second_player_net, second_cfg.planner)
    players = AdversarialMinds(first_player, second_player)

    # Create callbacks
    tournament = Tournament()

    # Get checkpoints paths
    all_checkpoints_paths = utils.get_checkpoints_for_game(
        checkpoints_dir, cfg.self_play["game"])

    # Reduce gap to play at least one game when there is more than one checkpoint
    if gap >= len(all_checkpoints_paths):
        gap = len(all_checkpoints_paths) - 1
        log.info("Gap is too big. Reduced to {}".format(gap))

    # Gather players ids and checkpoints paths for cross-play
    players_ids = []
    checkpoints_paths = []
    for idx in range(0, len(all_checkpoints_paths), gap):
        players_ids.append(idx)
        checkpoints_paths.append(all_checkpoints_paths[idx])

    # Create table for results, extra column for player id
    results = np.zeros(
        (len(checkpoints_paths), len(checkpoints_paths)), dtype=int)

    # Create ELO scoreboard
    elo = ELOScoreboard(players_ids)

    for i, (first_player_id, first_checkpoint_path) in enumerate(zip(players_ids, checkpoints_paths)):
        first_player_net.load_checkpoint(first_checkpoint_path)

        tournament_wins = tournament_draws = 0
        opponents_elo = []
        for j in range(i + 1, len(players_ids)):
            second_player_id, second_checkpoint_path = players_ids[j], checkpoints_paths[j]
            second_player_net.load_checkpoint(second_checkpoint_path)

            # Clear players tree
            first_player.clear_tree()
            second_player.clear_tree()

            hrl.loop(cfg.env, players, vision, policy='deterministic', n_episodes=2,
                     train_mode=False, name="{} vs {}".format(first_player_id, second_player_id),
                     callbacks=[tournament, cfg.env])

            wins, losses, draws = tournament.results

            # Book keeping
            tournament_wins += wins
            tournament_draws += draws

            results[i][j] = wins - losses
            results[j][i] = losses - wins

            opponents_elo.append(elo.scores.loc[second_player_id, 'elo'])

            # Update ELO rating of second player
            elo.update_player(second_player_id, elo.scores.loc[first_player_id, 'elo'],
                              losses, draws)

        # Update ELO rating of first player
        elo.update_player(first_player_id, opponents_elo,
                          tournament_wins, tournament_draws)

    # Save elo to csv
    elo.save_csv(cfg.logging['save_elo_scoreboard_path'])

    scoreboard = np.concatenate(
        (np.array(players_ids).reshape(-1, 1),
         results,
         np.sum(results, axis=1).reshape(-1, 1),
         elo.scores.elo.values.reshape(-1, 1).astype(np.int)),
        axis=1
    )
    tab = tabulate(scoreboard, headers=players_ids + ["sum", "elo"], tablefmt="fancy_grid")
    log.info("Results:\n{}".format(tab))
    for player_id, player_elo, checkpoint_path in zip(players_ids, elo.scores['elo'], checkpoints_paths):
        log.info("ITER: {:3}, ELO: {:4}, PATH: {}".format(
            player_id, int(player_elo), checkpoint_path))


if __name__ == "__main__":
    cli()
