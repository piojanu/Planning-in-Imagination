#!/usr/bin/env python3
import humblerl as hrl
import json
import logging as log
import numpy as np
import os
import utils
import click

from keras.callbacks import EarlyStopping, TensorBoard
from tabulate import tabulate

from algos.alphazero import Planner
from algos.human import HumanPlayer
from env import GameEnv
from nn import build_keras_nn, KerasNet
from callbacks import BasicStats, CSVSaverWrapper, Storage, Tournament, RenderCallback
from metrics import ELOScoreboard


@click.group()
@click.pass_context
@click.option('-c', '--config-path', type=click.File('r'),
              help="Path to configuration file (Default: config.json)", default="config.json")
@click.option('--debug/--no-debug', help="Enable debug logging (Default: False)", default=False)
def cli(ctx, config_path, debug):
    # Get and set up logger level and formatter
    log.basicConfig(level=log.DEBUG if debug else log.INFO, format="[%(levelname)s]: %(message)s")

    # Parse .json file with arguments
    params = json.loads(config_path.read())

    # Get specific modules params
    nn_params = params.get("neural_net", {})
    training_params = params.get("training", {})
    planner_params = params.get("planner", {})
    logging_params = params.get("logging", {})
    storage_params = params.get("storage", {})
    self_play_params = params.get("self_play", {})

    # Create environment and game model
    game_name = self_play_params.get('game', 'tictactoe')
    env = GameEnv(name=game_name)
    game = env.game

    # Create context
    ctx.obj = (nn_params, training_params, planner_params, logging_params,
               storage_params, self_play_params, env, game_name, game, debug)


@cli.command()
@click.pass_context
def self_play(ctx):
    """Train player by self-play, retraining from self-played frames and changing best player when
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

    nn_params, training_params, planner_params, logging_params, storage_params, self_play_params, env, game_name, game, debug_mode = ctx.obj

    # Get params for best model ckpt creation and update threshold
    save_folder = logging_params.get('save_checkpoint_folder', 'checkpoints')
    tensorboard_folder = logging_params.get('tensorboard_log_folder', './logs/tensorboard')
    update_threshold = self_play_params.get("update_threshold", 0.55)

    # Create Minds, current and best
    current_net = KerasNet(build_keras_nn(game, nn_params), training_params)
    best_net = KerasNet(build_keras_nn(game, nn_params), training_params)

    # Load best nn if available
    try:
        ckpt_path = utils.get_newest_ckpt_fname(save_folder)
        best_net.load_checkpoint(save_folder, ckpt_path)
        log.info("Best mind has loaded latest checkpoint: {}".format(
            utils.get_newest_ckpt_fname(save_folder)))
        global_epoch = utils.get_checkpoints_epoch(ckpt_path)
    except:
        log.info("No initial checkpoint, starting tabula rasa.")
        global_epoch = 0

    # Copy best nn weights to current nn that will be trained
    current_net.model.set_weights(best_net.model.get_weights())

    # Create players
    best_player = Planner(game, best_net, planner_params)
    current_player = Planner(game, current_net, planner_params)

    self_play_players = [
        best_player,
        best_player
    ]

    tournament_players = [
        current_player,
        best_player
    ]

    # Create callbacks, storage and tournament
    storage = Storage(game, storage_params)
    train_stats = CSVSaverWrapper(
        BasicStats(), logging_params.get('save_self_play_log_path', './logs/self-play.log'))
    tournament_stats = CSVSaverWrapper(
        Tournament(), logging_params.get('save_tournament_log_path', './logs/tournament.log'), True)

    # Load storage date from disk (path in config)
    storage.load()

    iter = 0
    max_iter = self_play_params.get("max_iter", -1)
    min_examples = self_play_params.get("min_examples", -1)
    while max_iter == -1 or iter < max_iter:
        iter_counter_str = "{:03d}/{:03d}".format(iter + 1, max_iter) if max_iter > 0 \
            else "{:03d}/inf".format(iter + 1)

        # SELF-PLAY - gather data using best nn
        hrl.loop(env, self_play_players,
                 policy='proportional', warmup=self_play_params.get('policy_warmup', 12),
                 debug_mode=debug_mode, n_episodes=self_play_params.get('n_self_plays', 100),
                 name="Self-play  " + iter_counter_str, verbose=2,
                 callbacks=[best_player, train_stats, storage])

        # Store gathered data
        storage.store()

        # Proceed to training only if threshold is fulfilled
        if len(storage.big_bag) <= min_examples:
            log.warn("Skip training, gather minimum {} training examples!".format(min_examples))
            continue

        # TRAINING - improve neural net
        trained_data = storage.big_bag
        boards_input, target_pis, target_values = list(zip(*trained_data))

        global_epoch = current_net.train(data=np.array(boards_input),
                                         targets=[np.array(target_pis), np.array(target_values)],
                                         initial_epoch=global_epoch,
                                         callbacks=[TensorBoard(log_dir=tensorboard_folder)])

        # ARENA - only the best will generate data!
        # Clear players tree
        current_player.clear_tree()
        best_player.clear_tree()

        hrl.loop(env, tournament_players,
                 policy='deterministic', alternate_players=True, train_mode=False,
                 debug_mode=debug_mode, n_episodes=self_play_params.get('n_tournaments', 20),
                 name="Tournament " + iter_counter_str, verbose=2,
                 callbacks=[tournament_stats])

        wins, losses, _ = tournament_stats.callback.results
        if wins > 0 and float(wins) / (wins + losses) > update_threshold:
            best_fname = "_".join(['self_play', game_name, str(global_epoch)]) + ".ckpt"
            log.info("New best player: {}".format(best_fname))

            # Save best and exchange weights
            current_net.save_checkpoint(save_folder, best_fname)
            best_net.model.set_weights(current_net.model.get_weights())

        # Increment iterator
        iter += 1


@cli.command()
@click.pass_context
@click.option('-ckpt', '--checkpoint', help="Path to NN checkpoint, if None then start fresh (Default: None)", type=click.Path(), default=None)
@click.option('-save', '--save_dir', help="Dir where to save NN checkpoint, if None then don't save (Default: None)", type=click.Path(), default=None)
@click.option('--tensorboard/--no-tensorboard', help="Enable tensorboard logging (Default: False)", default=False)
def train(ctx, checkpoint, save_dir, tensorboard):
    """Train NN from passed configuration."""

    nn_params, training_params, _, logging_params, storage_params, _, _, game_name, game, _ = ctx.obj

    # Get TensorBoard log dir
    tensorboard_folder = logging_params.get('tensorboard_log_folder', './logs/tensorboard')

    # Create Keras NN
    net = KerasNet(build_keras_nn(game, nn_params), training_params)

    # Load checkpoint nn if available
    global_epoch = 0
    if checkpoint:
        net.load_checkpoint(checkpoint)
        global_epoch = utils.get_checkpoints_epoch(checkpoint)
        log.info("Loaded checkpoint: {}".format(checkpoint))

    # Create TensorBoard logging callback if enabled
    callbacks = []
    if tensorboard:
        callbacks.append(TensorBoard(log_dir=tensorboard_folder))

    # Create storage and load data
    storage = Storage(game, storage_params)
    storage.load()

    # Prepare training data
    trained_data = storage.big_bag
    boards_input, target_pis, target_values = list(zip(*trained_data))

    # Run training
    global_epoch = net.train(data=np.array(boards_input),
                             targets=[np.array(target_pis), np.array(target_values)],
                             initial_epoch=global_epoch,
                             callbacks=callbacks)

    # Save model checkpoint if path passed
    if save_dir:
        save_fname = "_".join(["train", game_name, str(global_epoch)]) + ".ckpt"
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

    nn_params, training_params, _, storage_params, _, _, _, game, _ = ctx.obj

    # Get training params
    batch_size = training_params.get('batch_size', 32)
    epochs = training_params.get('epochs', 50)

    # Create storage and load data
    storage = Storage(game, storage_params)
    storage.load()

    # Prepare training data
    trained_data = storage.big_bag
    boards_input, target_pis, target_values = list(zip(*trained_data))

    data = np.array(boards_input)
    targets = [np.array(target_pis), np.array(target_values)]

    # Prepare search space
    space = []
    for k, v in nn_params.items():
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
            nn_params[k] = v

        # Build Keras neural net model
        model = build_keras_nn(game, nn_params)

        # Fit model
        history = model.fit(data, targets,
                            batch_size=batch_size,
                            epochs=epochs,
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
    nn_params, training_params, planner_params, _, _, _, env, game_name, game, debug_mode = ctx.obj

    # Create Minds, current and best
    first_player_net = KerasNet(build_keras_nn(game, nn_params), training_params)
    second_player_net = KerasNet(build_keras_nn(game, nn_params), training_params)

    first_player_net.load_checkpoint(first_model_path)
    second_player_net.load_checkpoint(second_model_path)

    tournament = Tournament()
    render_callback = RenderCallback(env, render)
    first_player = Planner(game, first_player_net, planner_params)
    second_player = Planner(game, second_player_net, planner_params)
    hrl.loop(env, [first_player, second_player], alternate_players=True, policy='deterministic',
             n_episodes=n_games, train_mode=False, debug_mode=debug_mode,
             name="Test  models: {} vs {}".format(first_model_path.split(
                 "/")[-1], second_model_path.split("/")[-1]),
             callbacks=[render_callback, tournament])

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
    nn_params, training_params, planner_params, _, _, _, env, game_name, game, _ = ctx.obj

    # Create Mind for NN oponnent
    first_player_net = KerasNet(build_keras_nn(game, nn_params), training_params)
    first_player_net.load_checkpoint(model_path)
    first_player = Planner(game, first_player_net, planner_params)

    human_player = HumanPlayer(game)

    render_callback = RenderCallback(env, render=True, fancy=True)

    tournament = Tournament()
    hrl.loop(env, [first_player, human_player], alternate_players=True,
             policy='deterministic', n_episodes=n_games, train_mode=False,
             name="Test models: {} vs HUMAN".format(model_path.split("/")[-1]),
             callbacks=[tournament, render_callback])

    log.info("{} vs HUMAN results: {}".format(model_path.split("/")[-1], tournament.results))


@cli.command()
@click.pass_context
@click.argument('checkpoints_dir', nargs=1, type=click.Path(exists=True))
@click.option('-g', '--gap', help="Gap between versions of best model (Default: 2)", default=2)
def cross_play(ctx, checkpoints_dir, gap):
    """Validate trained models. Best networks play with each other.

        Args:
            checkpoints_dir: (string): Path to checkpoints with models.
    """
    nn_params, training_params, planner_params, logging_params, _, _, env, game_name, game, _ = ctx.obj

    # Create players and their minds
    first_player_net = KerasNet(build_keras_nn(game, nn_params), training_params)
    second_player_net = KerasNet(build_keras_nn(game, nn_params), training_params)
    first_player = Planner(game, first_player_net, planner_params)
    second_player = Planner(game, second_player_net, planner_params)

    # Create callbacks
    tournament = Tournament()

    # Get checkpoints paths
    all_checkpoints_paths = utils.get_checkpoints_for_game(checkpoints_dir, game_name)

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
    results = np.zeros((len(checkpoints_paths), len(checkpoints_paths)), dtype=int)

    # Create ELO scoreboard
    elo = ELOScoreboard(players_ids)

    for i, (first_player_id, first_checkpoint_path) in enumerate(zip(players_ids, checkpoints_paths)):
        first_player_net.load_checkpoint(first_checkpoint_path)

        tournament_wins = tournament_draws = 0
        opponents_ids = []
        opponents_elo = []
        for j in range(i + 1, len(players_ids)):
            second_player_id, second_checkpoint_path = players_ids[j], checkpoints_paths[j]
            second_player_net.load_checkpoint(second_checkpoint_path)

            # Clear players tree
            first_player.clear_tree()
            second_player.clear_tree()

            hrl.loop(env, [first_player, second_player], alternate_players=True, policy='deterministic',
                     n_episodes=2, train_mode=False,
                     name="{} vs {}".format(first_player_id, second_player_id),
                     callbacks=[tournament])

            wins, losses, draws = tournament.results

            # Book keeping
            tournament_wins += wins
            tournament_draws += draws

            opponents_ids.append(second_player_id)
            opponents_elo.append(elo.scores.loc[second_player_id, 'elo'])

            results[i][j] = wins - losses
            results[j][i] = losses - wins

            # Update ELO rating of second player
            elo.update_rating(second_player_id, [first_player_id], losses, draws)

        # Update ELO rating of first player
        elo.update_rating(first_player_id, opponents_ids, tournament_wins,
                          tournament_draws, opponents_elo=opponents_elo)

    # Save elo to csv
    elo.save_csv(logging_params.get('save_elo_scoreboard_path', './logs/scoreboard.csv'))

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
