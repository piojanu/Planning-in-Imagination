#!/usr/bin/env python3
import humblerl as hrl
import json
import logging as log
import numpy as np
import utils
import click

from keras.callbacks import ModelCheckpoint
from tabulate import tabulate

from algos.alphazero import Planner
from env import GameEnv
from nn import build_keras_nn, KerasNet
from callbacks import BasicStats, CSVSaverWrapper, Storage, Tournament, RenderCallback

# Get and set up logger level and formatter
log.basicConfig(level=log.DEBUG, format="[%(levelname)s]: %(message)s")


@click.group()
@click.pass_context
@click.option('-c', '--config-path', type=click.File('r'),
              help="Path to configuration file (Default: config.json)", default="config.json")
def cli(ctx, config_path):
    # Parse .json file with arguments
    params = json.loads(config_path.read())

    # Get specific modules params
    nn_params = params.get("neural_net", {})
    training_params = params.get("training", {})
    planner_params = params.get("planner", {})
    storage_params = params.get("storage", {})
    self_play_params = params.get("self_play", {})

    # Create environment and game model
    game_name = self_play_params.get('game', 'tictactoe')
    env = GameEnv(name=game_name)
    game = env.game

    # Create context
    ctx.obj = (nn_params, training_params, planner_params,
               storage_params, self_play_params, env, game_name, game)


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
    nn_params, training_params, planner_params, storage_params, self_play_params, env, game_name, game = ctx.obj

    # Get params for best model ckpt creation and update threshold
    save_folder = self_play_params.get('save_checkpoint_folder', 'checkpoints')
    save_filename = self_play_params.get('save_checkpoint_filename', 'best')
    update_threshold = self_play_params.get("update_threshold", 0.55)

    # Create Minds, current and best
    current_net = KerasNet(build_keras_nn(game, nn_params), training_params)
    best_net = KerasNet(build_keras_nn(game, nn_params), training_params)

    # Load best nn if available
    try:
        best_net.load_checkpoint(save_folder, utils.get_newest_ckpt_fname(save_folder))
        log.info("Best mind has loaded latest checkpoint: {}".format(
            utils.get_newest_ckpt_fname(save_folder)))
    except:
        best_net.save_checkpoint(save_folder, utils.make_ckpt_fname(game_name, save_filename))
        log.info("Created initial checkpoint.")

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
    storage = Storage(storage_params)
    train_stats = CSVSaverWrapper(
        BasicStats(), self_play_params.get('save_self_play_log_path', './logs/self-play.log'))
    tournament_stats = CSVSaverWrapper(
        Tournament(), self_play_params.get('save_tournament_log_path', './logs/tournament.log'), True)

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
                 policy='deterministic', warmup=self_play_params.get('policy_warmup', 12),
                 n_episodes=self_play_params.get('n_self_plays', 100),
                 name="Self-play  " + iter_counter_str, verbose=2,
                 callbacks=[train_stats, storage])
        storage.store()

        # Proceed to training only if threshold is fulfilled
        if len(storage.big_bag) <= min_examples:
            log.warn("Skip training, gather minimum {} training examples!".format(min_examples))
            continue

        # TRAINING - improve neural net
        trained_data = storage.big_bag
        boards_input, target_pis, target_values = list(zip(*trained_data))

        current_net.load_checkpoint(save_folder, utils.get_newest_ckpt_fname(save_folder))
        current_net.train(data=np.array(boards_input), targets=[
                          np.array(target_pis), np.array(target_values)])

        # ARENA - only the best will remain!
        hrl.loop(env, tournament_players,
                 policy='deterministic', warmup=self_play_params.get('policy_warmup', 12), temperature=0.2,
                 alternate_players=True, train_mode=False,
                 n_episodes=self_play_params.get('n_tournaments', 20),
                 name="Tournament " + iter_counter_str, verbose=2,
                 callbacks=[tournament_stats])

        wins, losses, draws = tournament_stats.callback.results
        if wins + losses > 0 and float(wins) / (wins + losses) > update_threshold:
            best_fname = utils.make_ckpt_fname(game_name, save_filename)
            log.info("New best player: {}".format(best_fname))
            current_net.save_checkpoint(save_folder, best_fname)
            best_net.load_checkpoint(save_folder, best_fname)

        # Increment iterator
        iter += 1


@cli.command()
@click.pass_context
@click.option('-ckpt', '--checkpoint', help="Path to NN checkpoint, if None then start fresh (Default: None)", type=click.Path(), default=None)
@click.option('-best', '--best_save', help="Path where to save current best NN checkpoint, if None then don't save (Default: None)", type=click.Path(), default=None)
def train(ctx, checkpoint, best_save):
    """Train NN from passed configuration."""

    nn_params, training_params, _, storage_params, _, _, _, game = ctx.obj

    # Create Keras NN
    net = KerasNet(build_keras_nn(game, nn_params), training_params)

    # Load checkpoint nn if available
    if checkpoint:
        net.load_checkpoint(checkpoint)
        log.info("Loaded checkpoint: {}".format(checkpoint))

    # Create model checkpoint callback if path passed
    callbacks = []
    if best_save:
        callbacks.append(ModelCheckpoint(best_save, save_best_only=True, verbose=1))

    # Create storage and load data
    storage = Storage(storage_params)
    storage.load()

    # Prepare training data
    trained_data = storage.big_bag
    boards_input, target_pis, target_values = list(zip(*trained_data))

    # Run training
    net.train(data=np.array(boards_input),
              targets=[np.array(target_pis), np.array(target_values)],
              callbacks=callbacks)


@cli.command()
@click.pass_context
@click.argument('first_model_path', nargs=1, type=click.Path(exists=True))
@click.argument('second_model_path', nargs=1, type=click.Path(exists=True))
@click.option('--render/--no-render', help="Enable rendering game (Default: True)", default=True)
def clash(ctx, first_model_path, second_model_path, render):
    """Test two models. Play `n_games` between themselves.

        Args:
            first_model_path: (string): Path to first player model.
            second_model_path (string): Path to second player model.
    """
    nn_params, training_params, planner_params, _, _, env, game_name, game = ctx.obj

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
             n_episodes=2, train_mode=False,
             name="Test  models: {} vs {}".format(first_model_path.split(
                 "/")[-1], second_model_path.split("/")[-1]),
             callbacks=[render_callback, tournament])

    log.info("{} vs {} results: {}".format(first_model_path.split("/")
                                           [-1], second_model_path.split("/")[-1], tournament.results))


@cli.command()
@click.pass_context
@click.argument('checkpoints_dir', nargs=1, type=click.Path(exists=True))
@click.option('-g', '--gap', help="Gap between versions of best model (Default: 2)", default=2)
def cross_play(ctx, checkpoints_dir, gap):
    """Validate trained models. Best networks play with each other.

        Args:
            checkpoints_dir: (string): Path to checkpoints with models.
    """
    nn_params, training_params, planner_params, _, _, env, game_name, game = ctx.obj

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
    results = np.zeros((len(checkpoints_paths), 1 + len(checkpoints_paths)), dtype=int)

    for i, (first_player_id, first_checkpoint_path) in enumerate(zip(players_ids, checkpoints_paths)):
        first_player_net.load_checkpoint(first_checkpoint_path)
        results[i][0] = first_player_id

        for j, (second_player_id, second_checkpoint_path) in enumerate(zip(players_ids, checkpoints_paths)):
            if first_player_id != second_player_id:
                second_player_net.load_checkpoint(second_checkpoint_path)

                hrl.loop(env, [first_player, second_player], alternate_players=True, policy='deterministic',
                         n_episodes=2, train_mode=False,
                         name="{} vs {}".format(first_player_id, second_player_id),
                         callbacks=[tournament])

                wins, losses, _ = tournament.results
                results[i][j + 1] = wins - losses

    tab = tabulate(results, headers=players_ids, tablefmt="fancy_grid")
    log.info("results:\n{}".format(tab))


if __name__ == "__main__":
    cli()
