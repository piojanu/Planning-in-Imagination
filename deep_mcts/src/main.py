import humblerl as hrl
import numpy as np

from algos.value_function import build_keras_nn, Planner
from env import GameEnv
from nn import KerasNet
from storage import Storage
from tournament import Tournament
import json


def train(params={}):
    """Train player by self-play, retraining from self-played frames and changing best player when new trained player
    beats currently best player.

    Args:
        params (JSON dict): extra parameters
            * 'game' (string):                     game name (Default: tictactoe)
            * 'update_threshold' (float):          required threshold to be new best player (Default: 0.55)
            * 'max_iter' (int):                    number of train process iterations (Default: 10)
            * 'save_checkpoint_folder' (string):   folder to save best models (Default: "best_nets")
            * 'save_checkpoint_filename' (string): filename of best model (Default: "best_net")
            * 'n_self_plays' (int):                number of self played episodes (Default: 20)
            * 'n_tournaments' (int):               number of tournament episodes (Default: 10)

    """

    # Create environment and game model
    train_params = params.get("train", {})
    env = GameEnv(name=train_params.get('game', 'tictactoe'))
    game = env.game
    best_net = KerasNet(build_keras_nn(game), params.get("neural_net", {}))
    current_player_net = KerasNet(build_keras_nn(game), params.get("neural_net", {}))
    best_net_version = 0

    save_folder = train_params.get('save_checkpoint_folder', 'best_nets')
    save_filename = train_params.get('save_checkpoint_filename', 'best_net')
    update_threshold = train_params.get("update_threshold", 0.55)
    best_net.save_checkpoint(save_folder, save_filename + str(best_net_version))

    # Create storage and tournament callbacks
    storage = Storage(params.get("storage", {}))
    tournament = Tournament()

    # Create players
    best_player = Planner(game, best_net, params.get("planner", {}))
    current_player = Planner(game, current_player_net, params.get("planner", {}))

    self_play_players = [
        best_player,
        best_player
    ]
    tournament_players = [
        current_player,
        best_player
    ]

    max_iter = train_params.get("max_iter", 10)
    for iter in range(max_iter):
        # Create players

        print("\n<--- SELF-PLAY")
        hrl.loop(env, self_play_players, policy='stochastic', n_episodes=train_params.get('n_self_plays', 20),
                 name="Iter {:03d}/{:03d}".format(iter + 1, max_iter), callbacks=[storage])

        trained_data = storage.big_bag
        boards, _, targets = list(zip(*trained_data))

        print("\nRETRAINING")
        current_player_net.load_checkpoint(save_folder, save_filename + str(best_net_version))
        current_player_net.train(data=np.array(boards), targets=np.array(targets))
        print("\n")

        tournament.reset()
        hrl.loop(env, tournament_players, policy='stochastic', n_episodes=train_params.get('n_tournaments', 10),
                 name="ARENA", callbacks=[tournament])

        wins, losses, draws = tournament.get_results()
        print("Tournament results: ", tournament.get_results())
        if wins + losses > 0 and float(wins) / (wins + losses) > update_threshold:
            best_net_version += 1
            print("New best player, currently it is version {}".format(best_net_version))
            current_player_net.save_checkpoint(save_folder, save_filename + str(best_net_version))
            best_net.load_checkpoint(save_folder, save_filename + str(best_net_version))


def play(params={}):
    """Play without training.
    """


def main():
    # Parse .json file with arguments
    # TODO (pj): Implement .json config parsing.
    with open('config.json') as handle:
        params = json.loads(handle.read())

    train(params)


if __name__ == "__main__":
    main()
