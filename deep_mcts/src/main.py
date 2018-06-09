import humblerl as hrl

from algos.dummy import build_keras_nn, Planner
from env import GameEnv
from nn import KerasNet
from storage import Storage
from tournament import Tournament
from random import shuffle


def train(params={}):
    """Train player by self-play, retraining from self-played frames and changing best player when new trained player
    beats currently best player.

    Args:
        params (JSON dict): extra parameters
            * 'game' (string):                       game name (Default: tictactoe)
            * 'update_threshold' (float):          required threshold to be new best player (Default: 0.55)
            * 'max_iter' (int):                    number of train process iterations (Default: 10)
            * 'save_checkpoint_folder' (string):   folder to save best models (Default: "best_nets")
            * 'save_checkpoint_filename' (string): filename of best model (Default: "best_net")

    """

    # Create environment and game model
    env = GameEnv(name=params.get('game', 'tictactoe'))
    game = env.game
    best_net = KerasNet(build_keras_nn(game), params)
    current_player_net = KerasNet(build_keras_nn(), params)
    best_net_version = 0

    save_folder = params.get('save_checkpoint_folder', 'best_nets')
    save_filename = params.get('save_checkpoint_filename', 'best_net')
    update_threshold = params.get("update_threshold", 0.55)
    # TODO: (mj) UNCOMMENT WHEN MODEL WILL BE READY
    # best_net.save_checkpoint(save_folder, save_filename + str(best_net_version))

    # Create storage and tournament callbacks
    storage = Storage(params)
    tournament = Tournament()

    # Create players
    best_player = Planner(game, best_net, params)
    current_player = Planner(game, current_player_net, params)

    self_play_players = [
        best_player,
        best_player
    ]
    tournament_players = [
        current_player,
        best_player
    ]

    iter = 0
    max_iter = params.get("max_iter", 10)

    while iter < max_iter:
        # Create players

        print("---Epoch {:03d}/{:03d}---".format(iter + 1, max_iter))
        print("-------Self-play-------")
        hrl.loop(env, self_play_players, n_episodes=10, train_mode=True, callbacks=[storage])

        print("-------Retraining-------")
        trained_data = storage.big_bag
        shuffle(trained_data)

        # TODO: (mj) UNCOMMENT WHEN MODEL WILL BE READY
        # current_player_net.load_checkpoint(save_folder, save_filename + str(best_net_version))
        # current_player_net.train(data=trained_data[0], targets=trained_data[2])

        print("-------Tournament------")
        tournament.reset()
        hrl.loop(env, tournament_players, n_episodes=10, train_mode=False, callbacks=[tournament])

        print("--------Results--------")
        wins, losses, draws = tournament.get_results()
        print(tournament.get_results())
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
    params = {}
    train(params)


if __name__ == "__main__":
    main()
