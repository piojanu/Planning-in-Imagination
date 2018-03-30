import numpy as np


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
