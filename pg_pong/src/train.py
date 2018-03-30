import argparse

import gym
import numpy as np

import utils
from model import PolicyGradientModel


def parse_arguments():
    argparser = argparse.ArgumentParser(description='Script for training an agent to play Pong '
                                                    'game using the Policy Gradient method.')
    argparser.add_argument('-hs', '--hidden_size', type=int, default=50,
                           help='Number of neurons in hidden layer.')
    argparser.add_argument('-bs', '--batch_size', type=int, default=4,
                           help='Number of episodes in batch.')
    argparser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                           help='Learning rate.')
    argparser.add_argument('-d', '--discount', type=int, default=0.99,
                           help='Discount factor for reward.')
    argparser.add_argument('-r', '--render', action='store_true', default=False,
                           help='If True, Pong\'s board is displayed')
    return argparser.parse_args()


def main():
    args = parse_arguments()
    print(f'Arguments: {args}')
    env = gym.make('Pong-v0')
    pong_frame = env.reset()
    prev_state = None
    input_size = 6400
    num_actions = 3
    model = PolicyGradientModel(input_size=input_size, hidden_size=args.hidden_size,
                                output_size=num_actions)

    while True:
        if args.render:
            env.render()
        state = utils.preprocess_pong_state(pong_frame) - prev_state \
                if prev_state is not None else np.zeros(input_size)
        prev_state = state
        action = model.choose_action(state)
        pong_frame, reward, done, _ = env.step(action)
        if done:
            env.reset()
            prev_state = None


if __name__ == "__main__":
    main()
