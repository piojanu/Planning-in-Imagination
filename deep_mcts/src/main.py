import humblerl as hrl
import json
import logging as log
import numpy as np
import utils
import click

from algos.alphazero import build_keras_nn, Planner
from env import GameEnv
from nn import KerasNet
from callbacks import BasicStats, CSVSaverWrapper, Storage, Tournament, RenderCallback

# Get and set up logger level and formatter
log.basicConfig(level=log.DEBUG, format="[%(levelname)s]: %(message)s")


@click.group()
@click.pass_context
@click.option('-c', '--config-file', type=click.File('r'),
              help="Config file (Default: config.json)", default="config.json")
def main(context, config_file):
    # Parse .json file with arguments
    context.obj = json.loads(config_file.read())


@main.command()
@click.pass_context
def train(context={}):
    """Train player by self-play, retraining from self-played frames and changing best player when
    new trained player beats currently best player.

    Args:
        context (click.core.Context): context object.
            context.obj (JSON dict): configuration parameters
            Parameters for training:
                * 'game' (string)                     : game name (Default: tictactoe)
                * 'update_threshold' (float):         : required threshold to be new best player
                                                        (Default: 0.55)
                * 'policy_warmup' (int)               : how many stochastic warmup steps should take
                                                        deterministic policy (Default: 12)
                * 'max_iter' (int)                    : number of train process iterations
                                                        (Default: -1)
                * 'save_checkpoint_folder' (string)   : folder to save best models
                                                        (Default: "checkpoints")
                * 'save_checkpoint_filename' (string) : filename of best model (Default: "bestnet")
                * 'n_self_plays' (int)                : number of self played episodes
                                                        (Default: 100)
                * 'n_tournaments' (int)               : number of tournament episodes (Default: 20)

    """
    params = context.obj
    # Get params for different MCTS parts
    nn_params = params.get("neural_net", {})
    planner_params = params.get("planner", {})
    storage_params = params.get("storage", {})
    train_params = params.get("train", {})

    # Get params for best model ckpt creation and update threshold
    save_folder = train_params.get('save_checkpoint_folder', 'checkpoints')
    save_filename = train_params.get('save_checkpoint_filename', 'bestnet')
    update_threshold = train_params.get("update_threshold", 0.55)

    # Create environment and game model
    game_name = train_params.get('game', 'tictactoe')
    env = GameEnv(name=game_name)
    game = env.game

    # Create Minds, current and best
    current_net = KerasNet(build_keras_nn(game, nn_params), nn_params)
    best_net = KerasNet(build_keras_nn(game, nn_params), nn_params)

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
        BasicStats(), train_params.get('save_train_log_path', './logs/training.log'))
    tournament_stats = CSVSaverWrapper(
        Tournament(), train_params.get('save_tournament_log_path', './logs/tournament.log'), True)

    # Load storage date from disk (path in config)
    storage.load()

    iter = 0
    max_iter = train_params.get("max_iter", -1)
    while max_iter == -1 or iter < max_iter:
        iter_counter_str = "{:03d}/{:03d}".format(iter + 1, max_iter) if max_iter > 0 \
            else "{:03d}/inf".format(iter + 1)

        # SELF-PLAY - gather data using best nn
        hrl.loop(env, self_play_players,
                 policy='deterministic', warmup=train_params.get('policy_warmup', 12),
                 n_episodes=train_params.get('n_self_plays', 100),
                 name="Self-play  " + iter_counter_str,
                 callbacks=[train_stats, storage])
        storage.store()

        # TRAINING - improve neural net
        trained_data = storage.big_bag
        boards_input, target_pis, target_values = list(zip(*trained_data))

        current_net.load_checkpoint(save_folder, utils.get_newest_ckpt_fname(save_folder))
        current_net.train(data=np.array(boards_input), targets=[
                          np.array(target_pis), np.array(target_values)])

        # ARENA - only the best will remain!
        hrl.loop(env, tournament_players,
                 policy='deterministic', warmup=train_params.get('policy_warmup', 12), temperature=0.2,
                 alternate_players=True, train_mode=False,
                 n_episodes=train_params.get('n_tournaments', 20),
                 name="Tournament " + iter_counter_str,
                 callbacks=[tournament_stats])

        wins, losses, draws = tournament_stats.callback.results
        if wins + losses > 0 and float(wins) / (wins + losses) > update_threshold:
            best_fname = utils.make_ckpt_fname(game_name, save_filename)
            log.info("New best player: {}".format(best_fname))
            current_net.save_checkpoint(save_folder, best_fname)
            best_net.load_checkpoint(save_folder, best_fname)

        # Increment iterator
        iter += 1


@main.command()
@click.pass_context
def play(context):
    """Play without training."""
    # TODO (mj): Fill implementation and docstring


@main.command()
@click.pass_context
@click.argument('first_model_path', nargs=1, type=click.Path(exists=True))
@click.argument('second_model_path', nargs=1, type=click.Path(exists=True))
@click.option('--render/--no-render', help="Enable rendering game (Default: True)", default=True)
def test(context, first_model_path, second_model_path, render):
    """Test two models. Play `n_games` between themselves.

        Args:

            first_model_path: (string): Path to first player model.
            second_model_path (string): Path to second player model.
    """

    params = context.obj
    nn_params = params.get("neural_net", {})
    planner_params = params.get("planner", {})
    train_params = params.get("train", {})
    game_name = train_params.get('game', 'tictactoe')
    env = GameEnv(name=game_name)
    game = env.game

    # Create Minds, current and best
    first_player_net = KerasNet(build_keras_nn(game, nn_params), nn_params)
    second_player_net = KerasNet(build_keras_nn(game, nn_params), nn_params)

    first_player_net.load_checkpoint_from_path(first_model_path)
    second_player_net.load_checkpoint_from_path(second_model_path)

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


if __name__ == "__main__":
    main()
