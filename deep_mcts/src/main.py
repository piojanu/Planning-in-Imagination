import humblerl as hrl

from algos.dummy import build_keras_nn, Planner
from env import GameEnv
from nn import KerasNet
from storage import Storage
from tournament import Tournament


def main():
    # Parse .json file with arguments
    # TODO (pj): Implement .json config parsing.
    params = {}

    # Create environment, game, nn model
    env = GameEnv(name=params.get('game', 'connect4'))
    game = env.game
    nn = KerasNet(build_keras_nn(game), params)

    # Create storage and tournament callbacks
    storage = Storage(params)
    tournament = Tournament()

    # Create players
    players = [
        Planner(game, nn, params),
        Planner(game, nn, params)
    ]

    print("-------Self-play-------")
    hrl.loop(env, players, n_episodes=10, train_mode=True, callbacks=[storage])

    print("-------Tournament------")
    hrl.loop(env, players, n_episodes=10, train_mode=False, callbacks=[tournament])

    print("--------Results--------")
    print(tournament.get_results())


if __name__ == "__main__":
    main()
