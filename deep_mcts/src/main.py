import humblerl as hrl

from algos.dummy import Planner
from env import GameEnv
from storage import Storage


def main():
    # Parse .json file with arguments
    # TODO (pj): Implement .json config parsing.
    args = {
        'game': 'connect4'
    }

    # Create environment and game model
    env = GameEnv(name=args['game'])
    game = env.game
    storage = Storage()

    # Create players
    players = [
        Planner(game),
        Planner(game)
    ]

    hrl.loop(env, players, n_episodes=10, train_mode=False, callbacks=[storage])


if __name__ == "__main__":
    main()
