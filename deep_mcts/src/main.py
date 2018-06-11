import humblerl as hrl
import json
import logging as log
import numpy as np
import utils

from algos.value_function import build_keras_nn, Planner
from env import GameEnv
from nn import KerasNet
from storage import Storage
from tournament import Tournament

# Get and set up logger level and formatter
log.basicConfig(level=log.DEBUG, format="[%(levelname)s]: %(message)s")


def train(params={}):
    """Train player by self-play, retraining from self-played frames and changing best player when new trained player
    beats currently best player.

    Args:
        params (JSON dict): extra parameters
            * 'game' (string):                     game name (Default: tictactoe)
            * 'update_threshold' (float):          required threshold to be new best player (Default: 0.55)
            * 'max_iter' (int):                    number of train process iterations (Default: 10)
            * 'save_checkpoint_folder' (string):   folder to save best models (Default: "checkpoints")
            * 'save_checkpoint_filename' (string): filename of best model (Default: "bestnet")
            * 'n_self_plays' (int):                number of self played episodes (Default: 20)
            * 'n_tournaments' (int):               number of tournament episodes (Default: 10)

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
    current_net = KerasNet(build_keras_nn(game), nn_params)
    best_net = KerasNet(build_keras_nn(game), nn_params)

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
    tournament = Tournament()

    max_iter = train_params.get("max_iter", 10)
    for iter in range(max_iter):
        # Create players

        hrl.loop(env, self_play_players, policy='stochastic', n_episodes=train_params.get('n_self_plays', 20),
                 name="Self-play  {:03d}/{:03d}".format(iter + 1, max_iter), callbacks=[storage])

        trained_data = storage.big_bag
        boards, _, targets = list(zip(*trained_data))

        current_net.load_checkpoint(save_folder, utils.get_newest_ckpt_fname(save_folder))
        current_net.train(data=np.array(boards), targets=np.array(targets))

        tournament.reset()
        hrl.loop(env, tournament_players, policy='stochastic', n_episodes=train_params.get('n_tournaments', 10),
                 name="Tournament {:03d}/{:03d}".format(iter + 1, max_iter), callbacks=[tournament])

        wins, losses, draws = tournament.get_results()
        log.info("Tournament results: {}".format(tournament.get_results()))
        if wins + losses > 0 and float(wins) / (wins + losses) > update_threshold:
            best_fname = utils.make_ckpt_fname(game_name, save_filename)
            log.info("New best player: {}".format(best_fname))
            current_net.save_checkpoint(save_folder, best_fname)
            best_net.load_checkpoint(save_folder, best_fname)


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
