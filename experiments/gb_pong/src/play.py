import argparse
import time

import gym
import numpy as np

import utils
from model import PolicyGradientModel


def parse_arguments():
    argparser = argparse.ArgumentParser(description='Use trained Policy Gradient model to play Pong.')
    argparser.add_argument('-hs', '--hidden_size', type=int, default=50,
                           help='Number of neurons in hidden layer.')
    argparser.add_argument('-lm', '--load_model', help='Load model from path.', required=True)
    argparser.add_argument('-d', '--delay', help='Delay between frames', default=2e-2, type=float)
    return argparser.parse_args()


def main():
    args = parse_arguments()
    prev_state = None
    input_size = 6400
    num_actions = 3
    model = PolicyGradientModel(input_size=input_size, hidden_size=args.hidden_size,
                                output_size=num_actions)
    utils.load_model(model, model_path=args.load_model)

    env = gym.make('Pong-v0')
    pong_frame = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(args.delay)
        state = utils.preprocess_pong_state(pong_frame) - prev_state \
            if prev_state is not None else np.zeros(input_size)
        prev_state = state
        action = model.choose_action(state)
        pong_frame, _, done, _ = env.step(action)
    env.close()


if __name__ == "__main__":
    main()
