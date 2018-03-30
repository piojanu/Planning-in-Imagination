import argparse
import time

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
    return argparser.parse_args()


def update_model(optimizer, rewards, log_probs, discount_factor):
    rws = utils.discount_rewards(rewards, discount_factor)
    rws = torch.FloatTensor(rws)
    rws = (rws - rws.mean()) / (rws.std() + np.finfo(np.float32).eps)  # pylint: disable=E1101
    loss = []
    for log_prob, reward in zip(log_probs, rws):
        loss.append(-log_prob * reward)
    optimizer.zero_grad()
    loss = torch.cat(loss).sum()  # pylint: disable=E1101
    loss.backward()
    optimizer.step()


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
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)
    rewards = []
    log_probs = []
    running_reward = None
    episode_num = 0

    while True:
        if args.render:
            env.render()
        state = utils.preprocess_pong_state(pong_frame) - prev_state \
                if prev_state is not None else np.zeros(input_size)
        prev_state = state
        action, log_prob = model.choose_action(state)
        log_probs.append(log_prob)
        pong_frame, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            episode_num += 1
            episode_reward = np.sum(rewards)
            # discounted_rewards = utils.discount_rewards(rewards, args.discount)
            running_reward = episode_reward if running_reward is None \
                                            else 0.99*running_reward + 0.01*episode_reward
            print(f'Episode {episode_num} finished! Episode total reward: {episode_reward} '
                  f'Running mean: {running_reward:.3f}')

            if episode_num % args.batch_size == 0:
                print('--------------------------------------------------')
                print('Updating model\'s parameters!')
                start_time = time.time()
                update_model(optimizer, rewards, log_probs, args.discount)
                print('Finished updating model\'s parameters! Time: {:.3f}s'
                      .format(time.time() - start_time))
                print('--------------------------------------------------')

            env.reset()
            rewards = []
            log_probs = []
            prev_state = None


if __name__ == "__main__":
    main()
