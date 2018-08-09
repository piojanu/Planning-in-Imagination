import json
import os

import numpy as np
import torch


def preprocess_pong_state(pong_state):
    """Preprocess 210x160x3 Pong game state (frame) into 6400 float vector

    Args:
        pong_state (np.array): Raw frame of Pong game.

    Returns:
        np.array: 6400x1 float vector
    """
    image = pong_state[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return image.astype(np.float).ravel()


def discount_rewards(rewards, discount_factor):
    """Calculate discounted rewards.

    Args:
        rewards (list): Collected rewards from episode.
        discount_factor (float): Discount factor.

    Returns:
        list: Rewards discounted using discount factor.
    """
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for i, reward in reversed(list(enumerate(rewards))):
        if reward != 0:
            running_add = 0
        running_add = running_add * discount_factor + reward
        discounted_rewards[i] = running_add
    return discounted_rewards


def save_model(model, checkpoints_dir, episode_num, args):
    checkpoint_path = os.path.join(
        checkpoints_dir, f'model_{model.hidden_size}_{episode_num}.ckpt')
    config_path = os.path.join(checkpoints_dir, 'config.json')
    print(f'--> Saved model to {checkpoint_path}!')
    torch.save(model.state_dict(), checkpoint_path)
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
        print(f'--> Saved config to {config_path}!')


def load_model(model, model_path):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f'Loaded model from {model_path}!')
