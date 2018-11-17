import logging as log

import numpy as np
import torch
from torch.utils.data import Dataset


class EPNDataset(Dataset):
    """Dataset of sequential data to train MDN-RNN."""

    def __init__(self, storage, sequence_len, terminal_prob=0.5):
        """Initialize Expert Prediction Network PyTorch dataset.

        Args:
            storage (ExperienceStorage): Storage of played trajectories.
            sequence_len (int): Desired output sequence len.
            terminal_prob (float): Probability of sampling sequence that finishes with
                terminal state. (Default: 0.5)

        Note:
            Arrays should have the same size of the first dimension and their type should be the
            same as desired Tensor type.
        """

        assert 0 < terminal_prob and terminal_prob <= 1.0, "0 < terminal_prob <= 1.0"

        self.storage = storage
        self.sequence_len = sequence_len
        self.terminal_prob = terminal_prob

    def __getitem__(self, idx):
        """Get sequence at random starting position of given sequence length from episode `idx`."""

        offset = 1

        tau = self.storage.big_bag[idx]
        states, actions, rewards, done_flags, pis, values = list(zip(*tau))
        episode_length = len(tau)

        if self.sequence_len <= episode_length - offset:
            sequence_len = self.sequence_len
        else:
            sequence_len = episode_length - offset
            log.warning(
                "Episode %d is too short to form full sequence, data will be zero-padded.", idx)

        # Sample where to start sequence of length `self.sequence_len` in episode `idx`
        # '- offset' because "next states" are offset by 'offset'
        if np.random.rand() < self.terminal_prob:
            # Take sequence ending with terminal state
            start = episode_length - sequence_len - offset
        else:
            # NOTE: np.random.randint takes EXCLUSIVE upper bound of range to sample from
            start = np.random.randint(max(1, episode_length - sequence_len - offset))

        # NOTE: All of those types are assumptions, if it'll be problem, change it.
        input_states = torch.FloatTensor(states[start:start + sequence_len])
        input_actions = torch.unsqueeze(torch.LongTensor(actions[start:start + sequence_len]), 1)

        target_states = torch.FloatTensor(states[start + offset:start + sequence_len + offset])
        target_rewards = torch.FloatTensor(rewards[start:start + sequence_len])
        target_dones = torch.FloatTensor(done_flags[start:start + sequence_len])
        target_pis = torch.FloatTensor(pis[start:start + sequence_len])
        target_values = torch.FloatTensor(values[start:start + sequence_len])

        return ([input_states, input_actions],
                [target_states, target_rewards, target_dones, target_pis, target_values])

    def __len__(self):
        return len(self.storage.big_bag)
