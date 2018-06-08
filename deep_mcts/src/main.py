import humblerl as hrl

from algos.dummy import EvaluationNet, Planner
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
    nn = EvaluationNet()
    storage = Storage(params)
    tournament = Tournament()

    # Build neural network
    nn.build(params)

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
