import humblerl as hrl

from algos.dummy import Planner
from env import GameEnv
from storage import Storage
from tournament import Tournament


def main():
    # Parse .json file with arguments
    # TODO (pj): Implement .json config parsing.
    params = {}

    # Create environment and game model
    env = GameEnv(name=params.get('game', 'connect4'))
    game = env.game

    storage = Storage(params)
    tournament = Tournament()

    # Create players
    players = [
        Planner(game, None, params),
        Planner(game, None, params)
    ]

    hrl.loop(env, players, n_episodes=10, train_mode=False, callbacks=[storage])

    print("-------Tournament------")

    hrl.loop(env, players, n_episodes=10, train_mode=True, callbacks=[tournament])
    print("--------Results--------")
    print(tournament.get_results())


if __name__ == "__main__":
    main()
