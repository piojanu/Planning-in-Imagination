import humblerl as hrl

from algos.dummy import Planner
from env import GameEnv
from storage import Storage


def main():
    # Parse .json file with arguments
    # TODO (pj): Implement .json config parsing.
    params = {}

    # Create environment and game model
    env = GameEnv(name=params.get('game', 'connect4'))
    game = env.game
    storage = Storage(params)

    # Create players
    players = [
        Planner(game, None, params),
        Planner(game, None, params)
    ]

    hrl.loop(env, players, n_episodes=10, train_mode=False, callbacks=[storage])


if __name__ == "__main__":
    main()
