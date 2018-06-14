import humblerl as hrl
import json
import logging as log
import numpy as np
import utils

from algos.alphazero import build_keras_nn, Planner
from env import GameEnv
from nn import KerasNet
from callbacks import BasicStats, Storage, Tournament

# Get and set up logger level and formatter
log.basicConfig(level=log.DEBUG, format="[%(levelname)s]: %(message)s")


def train(params={}):
    """Train player by self-play, retraining from self-played frames and changing best player when new trained player
    beats currently best player.

    Args:
        params (JSON dict): extra parameters
            * 'game' (string):                     game name (Default: tictactoe)
            * 'update_threshold' (float):          required threshold to be new best player (Default: 0.55)
            * 'max_iter' (int):                    number of train process iterations (Default: -1)
            * 'save_checkpoint_folder' (string):   folder to save best models (Default: "checkpoints")
            * 'save_checkpoint_filename' (string): filename of best model (Default: "bestnet")
            * 'n_self_plays' (int):                number of self played episodes (Default: 100)
            * 'n_tournaments' (int):               number of tournament episodes (Default: 20)

    """

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
    basicstats = BasicStats()
    storage = Storage(storage_params)
    tournament = Tournament()

    # Load storage date from disk (path in config)
    storage.load()

    iter = 0
    max_iter = train_params.get("max_iter", -1)
    while max_iter == -1 or iter < max_iter:
        iter_counter_str = "{:03d}/{:03d}".format(iter + 1, max_iter) if max_iter > 0 \
            else "{:03d}/inf".format(iter + 1)

        # SELF-PLAY - gather data using best nn
        hrl.loop(env, self_play_players, policy='deterministic', warmup=10,
                 n_episodes=train_params.get('n_self_plays', 100),
                 name="Self-play  " + iter_counter_str,
                 callbacks=[basicstats, storage])
        storage.store()

        # TRAINING - improve neural net
        trained_data = storage.big_bag
        boards_input, target_pis, target_values = list(zip(*trained_data))

        current_net.load_checkpoint(save_folder, utils.get_newest_ckpt_fname(save_folder))
        current_net.train(data=np.array(boards_input), targets=[
                          np.array(target_pis), np.array(target_values)])

        # ARENA - only the best will remain!
        tournament.reset()
        hrl.loop(env, tournament_players, alternate_players=True, policy='deterministic', warmup=5,
                 n_episodes=train_params.get('n_tournaments', 20),
                 name="Tournament " + iter_counter_str,
                 callbacks=[tournament])

        wins, losses, draws = tournament.results
        if wins + losses > 0 and float(wins) / (wins + losses) > update_threshold:
            best_fname = utils.make_ckpt_fname(game_name, save_filename)
            log.info("New best player: {}".format(best_fname))
            current_net.save_checkpoint(save_folder, best_fname)
            best_net.load_checkpoint(save_folder, best_fname)

        # Increment iterator
        iter += 1


def play(params={}):
    """Play without training."""


def main():
    # Parse .json file with arguments
    with open('config.json') as handle:
        params = json.loads(handle.read())

    # Train!
    train(params)


if __name__ == "__main__":
    main()
