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


class EPNVision(Vision, Callback):
    """Expert Prediction Network vision processors.

    Args:
        vae_model (keras.Model): Keras VAE encoder.
        epn_model (torch.nn.Module): PyTorch EPN-RNN memory module.

    Note:
        In order to work, this Vision system must be also passed as callback to 'hrl.loop(...)'!
    """

    def __init__(self, vae_model, epn_model):
        self.vae_model = vae_model
        self.epn_model = epn_model

    def __call__(self, state, reward=0.):
        # WARN: See HRL `ply`, `on_step_taken` that would update hidden state is called AFTER
        #       Vision is used to preprocess next_state. So next_state has out-dated hidden state!
        #       What saves us is the fact, that `state` in next `ply` call will have it updated so,
        #       Transitions.state has up-to-date latent and hidden state and in all the other places
        #       exactly it is used, not next state.
        return self.process_state(state), reward

    def process_state(self, state):
        # NOTE: [0][0] <- it gets first in the batch latent space mean (mu)
        latent = self.vae_model.predict(state[np.newaxis, :])[0][0]
        hidden = self.epn_model.hidden

        # Agent never get terminal state to decide on. Always return False as done flag of state.
        return (latent, hidden, False)

    def on_episode_start(self, episode, train_mode):
        self.epn_model.init_hidden(1)

    def on_step_taken(self, step, transition, info):
        device = next(self.epn_model.parameters()).device

        latent = torch.as_tensor(
            transition.state[0], dtype=torch.float, device=device).view(1, 1, -1)
        hidden = transition.state[1]
        action = torch.as_tensor(
            transition.action, dtype=torch.long, device=device).view(1, 1, -1)

        with torch.no_grad(), evaluate(self.epn_model) as net:
            net(latent, action, hidden)


class EPNDataset(Dataset):
    """Dataset of sequential data to train EPN-RNN.

    Args:
        storage (ExperienceStorage): Storage of played trajectories.
        sequence_len (int): Desired output sequence len.
        terminal_prob (float): Probability of sampling sequence that finishes with
            terminal state. (Default: 0.5)

    Note:
        Arrays should have the same size of the first dimension and their type should be the
        same as desired Tensor type.
    """

    def __init__(self, storage, sequence_len, terminal_prob=0.5):
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
        target_rewards = torch.FloatTensor(rewards[start:start + sequence_len]).view(-1, 1)
        target_dones = torch.FloatTensor(done_flags[start:start + sequence_len]).view(-1, 1)
        target_pis = torch.FloatTensor(pis[start:start + sequence_len])
        target_values = torch.FloatTensor(values[start:start + sequence_len]).view(-1, 1)

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

    def sample(self, latent, hidden, action):
        """Predicts next state, reward, if done and expert pi and value.

        Args:
            latent (np.array): Latent state vector.
            hidden (torch.Tensor): EPN hidden state.
            action (int): Action to take in provided state.

        Returns:
            np.array: Next latent state vector.
            torch.Tensor: Next EPN hidden state.
            float: Transition reward.
            bool: If transition is terminal.
        """

        device = next(self.parameters()).device

        latent = torch.as_tensor(latent, dtype=torch.float, device=device).view(1, 1, -1)
        action = torch.tensor(action, dtype=torch.long, device=device).view(1, 1, -1)

        with torch.no_grad(), evaluate(self) as net:
            next_state, reward, done, _, _ = net(latent, action, hidden).values()

        return (torch.squeeze(next_state).cpu().detach().numpy(),
                self.hidden,
                torch.squeeze(reward).item(),
                torch.squeeze(done).item())

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device

        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        )


class SokobanMDP(MDP):
    """Simplified Sokoban MDP using memory module as dynamics model.

    Args:
        env (hrl.Environment): Environment which this MDP model.
        epn_model (torch.nn.Module): PyTorch EPN-RNN memory.

    Note:
        EPN-RNN memory module will change its internal hidden state each time `transition` is
        called. Remember about it.
    """

    def __init__(self, env, epn_model, done_threshold):
        self.env = env
        self.epn_model = epn_model
        self.done_threshold = done_threshold

    def transition(self, state, action):
        """Perform `action` in `state`. Return outcome.

        Args:
            state (tuple): MDP's state (observation latent vector, memory hidden state, if done).
            action (int): MDP's action.

        Returns:
            state (tuple): MDP's next state (observation latent vector, memory hidden state, if done).
            float: Reward.
        """

        latent, hidden, is_done = state
        if is_done:
            return state

        next_latent, next_hidden, reward, is_done_prob = \
            self.epn_model.sample(latent, hidden, action)

        # Determine if this terminal state
        is_done = is_done_prob > self.done_threshold

        # If terminal state, then determine if won or lost
        result = 0
        if is_done:
            result = 1 if reward > 0 else -1

        # TODO: This should predict reward
        return (next_latent, next_hidden, is_done), result

    def get_init_state(self):
        """Prepare and return initial state.

        It's not needed and left not implemented.
        """

        # NOTE: It should take env init state, encode it and return in tuple with zero hidden state,
        #       but it's left not implemented as long as it's not used anywhere.
        raise NotImplementedError()

    def get_valid_actions(self, state):
        """Get available actions in `state`.

        Args:
            state (tuple): MDP's state (observation latent vector, memory hidden state).

        Returns:
            np.ndarray: Array with enumerated valid actions for given state.
        """

        assert self.env.is_discrete, "This MDP works only for discrete action space!"
        # In OpenAI Gym all of the actions are always valid.
        return self.env.valid_actions

    def is_terminal_state(self, state):
        """Check if `state` is terminal.

        Args:
            state (tuple): MDP's state (observation latent vector, memory hidden state).

        Returns:
            bool: Whether state is terminal or not.
        """

        return state[2]

    def action_space(self):
        """Get action space definition.

        Returns:
            ActionSpace: Action space, discrete or continuous.
        """

        return self.env.action_space

    def state_space(self):
        """Get environment state space definition.

        Returns:
            object: State space representation depends on model.
        """

        return self.env.state_space


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
