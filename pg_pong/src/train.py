import argparse
import os
import time
import datetime

import gym
import numpy as np
import torch

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
    argparser.add_argument('-sf', '--save_freq', type=int, default=50,
                           help='Frequency (in episodes) of model checkpoints.')
    argparser.add_argument('-o', '--output_dir', default='checkpoints',
                           help='Output directory for model checkpoints.')
    return argparser.parse_args()


def prepare_checkpoints_dir(output_dir):
    checkpoints_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    return checkpoints_dir


def update_model(optimizer):
    print('--------------------------------------------------')
    print('Updating model\'s parameters!')
    start_time = time.time()
    optimizer.step()
    optimizer.zero_grad()
    print('Finished updating model\'s parameters! Time: {:.3f}s'.format(time.time() - start_time))
    print('--------------------------------------------------')


def save_model(model, checkpoints_dir, episode_num):
    checkpoint_path = os.path.join(checkpoints_dir, f'model_{model.hidden_size}_{episode_num}.ckpt')
    print(f'--> Saving model to {checkpoint_path}!')
    torch.save(model.state_dict(), checkpoint_path)


def main():
    args = parse_arguments()
    checkpoints_dir = prepare_checkpoints_dir(args.output_dir)
    print(f'Arguments: {args}')
    print(f'Checkpoints dir: {checkpoints_dir}')
    env = gym.make('Pong-v0')
    pong_frame = env.reset()
    prev_state = None
    input_size = 6400
    num_actions = 3
    model = PolicyGradientModel(input_size=input_size, hidden_size=args.hidden_size,
                                output_size=num_actions)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    optimizer.zero_grad()
    running_reward = None
    episode_num = 0
    episode_reward = 0

    while True:
        if args.render:
            env.render()
        state = utils.preprocess_pong_state(pong_frame) - prev_state \
                if prev_state is not None else np.zeros(input_size)
        prev_state = state
        action = model.choose_action(state)
        pong_frame, reward, done, _ = env.step(action)
        model.rewards.append(reward)
        episode_reward += reward
        if done:
            episode_num += 1
            running_reward = episode_reward if running_reward is None \
                                            else 0.99*running_reward + 0.01*episode_reward
            print(f'Episode {episode_num} finished! Episode total reward: {episode_reward} '
                  f'Running mean: {running_reward:.3f}')

            model.backward(args.discount)
            # model.show_grads()

            if episode_num % args.batch_size == 0:
                update_model(optimizer)

            if episode_num % args.save_freq == 0:
                save_model(model, checkpoints_dir, episode_num)

            env.reset()
            prev_state = None
            episode_reward = 0


if __name__ == "__main__":
    main()
