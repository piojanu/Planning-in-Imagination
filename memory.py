import logging as log

from collections import OrderedDict
import numpy as np
import humblerl as hrl
from humblerl import Callback, MDP, Vision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from common_utils import get_model_path_if_exists
from world_models.third_party.torchtrainer import TorchTrainer, evaluate


class EPNDataset(Dataset):
    """Dataset of sequential data to train EPN-RNN."""

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


class EPN(nn.Module):
    def __init__(self, hidden_units, latent_dim, action_space, num_layers=1):
        super(EPN, self).__init__()

        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.embedding = nn.Embedding.from_pretrained(torch.eye(action_space.num)) \
            if isinstance(action_space, hrl.environments.Discrete) else None
        self.lstm = nn.LSTM(input_size=(latent_dim + action_space.num),
                            hidden_size=hidden_units,
                            num_layers=num_layers,
                            batch_first=True)

        self.next_state = nn.Linear(hidden_units, latent_dim)
        self.reward = nn.Linear(hidden_units, 1)
        self.done = nn.Linear(hidden_units, 1)
        self.pi = nn.Linear(hidden_units, action_space.num)
        self.value = nn.Linear(hidden_units, 1)

    def forward(self, latent, action, hidden=None):
        self.lstm.flatten_parameters()
        if self.embedding:
            # Use one-hot representation for discrete actions
            x = torch.cat((latent, self.embedding(action).squeeze(dim=2)), dim=2)
        else:
            # Pass raw action vector for continuous actions
            x = torch.cat((latent, action.float()), dim=2)

        h, self.hidden = self.lstm(x, hidden if hidden else self.hidden)

        next_state = self.next_state(h)
        reward = torch.tanh(self.reward(h))  # NOTE: For won/lost rewards!
        done = torch.tanh(self.done(h))
        pi = self.pi(h)
        value = torch.tanh(self.value(h))    # NOTE: For won/lost rewards!

        return OrderedDict([('next_state', next_state),
                            ('reward', reward),
                            ('done', done),
                            ('pi', pi),
                            ('value', value)])

    def simulate(self, latent, actions, hidden=None):
        """Simulate environment trajectory.

        Args:
            latent (torch.Tensor): Latent vector with state(s) to start from.
                Shape of tensor: batch x 1 (sequence dim.) x latent dim.
            actions (torch.Tensor): Tensor with actions to take in simulated trajectory.
                Shape of tensor: batch x sequence x action dim.
            hidden (tuple): Memory module (torch.nn.LSTM) hidden state.

        Return:
            np.ndarray: Array of latent vectors of simulated trajectory.
                Shape of array: batch x sequence x latent dim.

        Note:
            You can find next hidden state in this module `hidden` member.
        """

        states = []
        for a in range(actions.shape[1]):
            # NOTE: We use np.newaxis to preserve shape of tensor.
            with torch.no_grad(), evaluate(self) as net:
                latent, _, _, _, _ = net(latent, actions[:, a, np.newaxis, :], hidden).values()
                states.append(latent.cpu().detach().numpy())

        # TODO: Check if this squeeze is even needed.
        return np.transpose(np.squeeze(np.array(states), axis=2), axes=[1, 0, 2])

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device

        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        )


def build_rnn_model(rnn_params, latent_dim, action_space, model_path=None):
    """Builds EPN-RNN memory module, which model time dependencies.

    Args:
        rnn_params (dict): EPN-RNN parameters from .json config.
        latent_dim (int): Latent space dimensionality.
        action_space (hrl.environments.ActionSpace): Action space, discrete or continuous.
        model_path (str): Path to VAE ckpt. Taken from .json config if `None` (Default: None)

    Returns:
        TorchTrainer: Compiled EPN-RNN model wrapped in TorchTrainer, ready for training.
    """

    use_cuda = torch.cuda.is_available()

    def masked_reward_loss_function(pred, target):
        """Mask zero rewards. It assumes 1/0 won/lost rewards!"""

        pred = pred * torch.abs(target)
        return F.mse_loss(pred, target)

    def cross_entropy_loss_function(pred, target):
        """Cross entropy that accepts soft targets."""

        return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1))

    epn = TorchTrainer(EPN(rnn_params['hidden_units'], latent_dim, action_space),
                       device_name='cuda' if use_cuda else 'cpu')

    epn.compile(
        optimizer=optim.Adam(epn.model.parameters(), lr=rnn_params['learning_rate']),
        loss={
            'next_state': nn.MSELoss(),
            'reward': masked_reward_loss_function,
            'done': nn.MSELoss(),
            'pi': cross_entropy_loss_function,
            'value': nn.MSELoss()
        }
    )

    model_path = get_model_path_if_exists(
        path=model_path, default_path=rnn_params['ckpt_path'], model_name="EPN-RNN")

    if model_path is not None:
        epn.load_ckpt(model_path)
        log.info("Loaded EPN-RNN model weights from: %s", model_path)

    return epn
