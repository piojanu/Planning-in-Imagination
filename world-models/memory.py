import logging as log


import numpy as np
import os.path
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import humblerl as hrl

from humblerl import Callback, Vision
from third_party.torchtrainer import TorchTrainer
from torch.distributions import Normal
from torch.utils.data import Dataset

from utils import get_model_path_if_exists


class MDNVision(Vision, Callback):
    def __init__(self, vae_model, mdn_model, latent_dim, state_processor_fn):
        """Initialize vision processors.

        Args:
            vae_model (keras.Model): Keras VAE encoder.
            mdn_model (torch.nn.Module): PyTorch MDN-RNN memory.
            latent_dim (int): Latent space dimensionality.
            state_processor_fn (function): Function for state processing. It should
                take raw environment state as an input and return processed state.

        Note:
            In order to work, this Vision system must be also passed as callback to 'hrl.loop(...)'!
        """

        self.vae_model = vae_model
        self.mdn_model = mdn_model
        self.latent_dim = latent_dim
        self.state_processor_fn = state_processor_fn

    def __call__(self, state, reward=0.):
        return self.process_state(state), reward

    def process_state(self, state):
        # NOTE: [0][0] <- it gets first in the batch latent space mean (mu)
        latent = self.vae_model.predict(self.state_processor_fn(state)[np.newaxis, :])[0][0]
        memory = self.mdn_model.hidden[0].cpu().detach().numpy()

        return np.concatenate((latent, memory.flatten()))

    def on_episode_start(self, episode, train_mode):
        self.mdn_model.init_hidden(1)

    def on_step_taken(self, step, transition, info):
        state = torch.from_numpy(transition.state[:self.latent_dim]).view(1, 1, -1)
        action = torch.from_numpy(np.array([transition.action])).view(1, 1, -1)
        if torch.cuda.is_available():
            state = state.cuda()
            action = action.cuda()
        self.mdn_model(state, action)


class StoreMemTransitions(Callback):
    """Save transitions for Memory training to HDF5 file in four datasets:
        * 'states': States preprocessed by MDNVision.
        * 'actions': Actions.
        * 'rewards': Rewards.
        * 'episodes': Indices of each episode (episodes[i] -> start index of episode `i`
                      in states and actions datasets).

        Datasets are organized in such a way, that you can locate episode `i` by accessing
        i-th position in `episodes` to get the `start` index and (i+1)-th position to get
        the `end` index and then get all of this episode's transitions by accessing
        `states[start:end]` and `actions[start:end]`.

        HDF5 file also keeps meta-informations (attributes) as such:
        * 'N_TRANSITIONS': Datasets size (number of transitions).
        * 'N_GAMES': From how many games those transitions come from.
        * 'LATENT_DIM': VAE's latent state dimensionality.
        * 'ACTION_DIM': Action's dimensionality (1 for discrete).
    """

    def __init__(self, out_path, latent_dim, action_space, min_transitions=10000, min_episodes=1000, chunk_size=128):
        """Initialize memory data storage.

        Args:
            out_path (str): Path to output hdf5 file.
            latent_dim (int): VAE's latent state dimensionality.
            action_space (hrl.environments.ActionSpace): Object representing action space,
                check HumbleRL.
            min_transitions (int): Minimum expected number of transitions in dataset. If more is
                gathered, then hdf5 dataset size is expanded.
            min_episodes (int): Minimum expected number of episodes in dataset. If more is
                gathered, then hdf5 dataset size is expanded.
            chunk_size (int): Chunk size in transitions. For efficiency reasons, data is saved
                to file in chunks to limit the disk usage (chunk is smallest unit that get fetched
                from disk). For best performance set it to training batch size. (Default: 128)
        """

        self.out_path = out_path
        self.dataset_size = min_transitions
        self.min_transitions = min_transitions
        self.episodes_size = min_episodes
        self.latent_dim = latent_dim
        self.action_dim = action_space.num if isinstance(
            action_space, hrl.environments.Continuous) else 1
        self.transition_count = 0
        self.game_count = 0

        # Make sure that path to out file exists
        dirname = os.path.dirname(out_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        # Create output hdf5 file and fill metadata
        self.out_file = h5py.File(out_path, "w")
        self.out_file.attrs["N_TRANSITIONS"] = 0
        self.out_file.attrs["N_GAMES"] = 0
        self.out_file.attrs["CHUNK_SIZE"] = chunk_size
        self.out_file.attrs["LATENT_DIM"] = latent_dim
        self.out_file.attrs["ACTION_DIM"] = self.action_dim

        # Create datasets
        self.out_states = self.out_file.create_dataset(
            name="states", dtype=np.float32, chunks=(chunk_size, 2, latent_dim),
            shape=(self.dataset_size, 2, latent_dim), maxshape=(None, 2, latent_dim),
            compression="lzf")
        self.out_actions = self.out_file.create_dataset(
            name="actions", dtype=action_space.sample().dtype, chunks=(chunk_size, self.action_dim),
            shape=(self.dataset_size, self.action_dim), maxshape=(None, self.action_dim),
            compression="lzf")
        self.out_rewards = self.out_file.create_dataset(
            name="rewards", dtype=np.float32, chunks=(chunk_size,),
            shape=(self.dataset_size,), maxshape=(None,),
            compression="lzf")
        self.out_episodes = self.out_file.create_dataset(
            name="episodes", dtype=np.int, chunks=(chunk_size,),
            shape=(self.episodes_size + 1,), maxshape=(None,))

        self.states = []
        self.actions = []
        self.rewards = []
        self.out_episodes[0] = 0

    def on_step_taken(self, step, transition, info):
        action = transition.action
        self.states.append(transition.state)
        self.actions.append(action if isinstance(action, np.ndarray) else [action])
        self.rewards.append(transition.reward)

        self.transition_count += 1

        if transition.is_terminal:
            self.game_count += 1
            if self.game_count == self.episodes_size:
                self.episodes_size *= 2
                self.out_episodes.resize(self.episodes_size, axis=0)
            self.out_episodes[self.game_count] = self.transition_count

        if self.transition_count % self.min_transitions == 0:
            self._save_chunk()

    def on_loop_end(self, is_aborted):
        if len(self.states) > 0:
            self._save_chunk()

        # Close file
        self.out_file.close()

    def _save_chunk(self):
        """Save `states` and `actions` to HDF5 file. Clear the buffers.
        Update transition and games count in HDF5 file."""

        # Resize datasets if needed
        if self.transition_count > self.dataset_size:
            self.out_states.resize(self.transition_count, axis=0)
            self.out_actions.resize(self.transition_count, axis=0)
            self.out_rewards.resize(self.transition_count, axis=0)
            self.dataset_size = self.transition_count

        n_transitions = len(self.states)
        start = self.transition_count - n_transitions

        assert n_transitions > 0, "Nothing to save!"

        self.out_states[start:self.transition_count] = self.states
        self.out_actions[start:self.transition_count] = self.actions
        self.out_rewards[start:self.transition_count] = self.rewards

        self.out_file.attrs["N_TRANSITIONS"] = self.transition_count
        self.out_file.attrs["N_GAMES"] = self.game_count

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()


class MDNDataset(Dataset):
    """Dataset of sequential data to train MDN-RNN."""

    def __init__(self, dataset_path, sequence_len):
        """Initialize MDNDataset.

        Args:
            sequence_len (int): Desired output sequence len.

        Note:
            Arrays should have the same size of the first dimension and their type should be the
            same as desired Tensor type.
        """

        self.dataset = self.out_file = h5py.File(dataset_path, "r")
        self.sequence_len = sequence_len
        self.latent_dim = self.dataset.attrs["LATENT_DIM"]
        self.action_dim = self.dataset.attrs["ACTION_DIM"]

    def __getitem__(self, idx):
        """Get sequence at random starting position of given sequence length from episode `idx`."""

        offset = 1

        t_start, t_end = self.dataset['episodes'][idx:idx + 2]
        episode_length = t_end - t_start
        # Sample where to start sequence of length `self.sequence_len` in episode `idx`
        # '- offset' because "next states" are offset by 'offset'
        sequence_len = self.sequence_len if self.sequence_len <= episode_length - offset \
            else episode_length - offset
        # NOTE: np.random.randint takes EXCLUSIVE upper bound of range to sample from, that's why
        #       one is added.
        start = t_start + np.random.randint(episode_length - sequence_len - offset + 1)

        states_ = torch.from_numpy(self.dataset['states'][start:start + sequence_len + offset])
        actions_ = torch.from_numpy(self.dataset['actions'][start:start + sequence_len])

        states = torch.zeros(self.sequence_len, self.latent_dim, dtype=states_.dtype)
        next_states = torch.zeros(self.sequence_len, self.latent_dim, dtype=states_.dtype)
        actions = torch.zeros(self.sequence_len, self.action_dim, dtype=actions_.dtype)

        # Sample latent states (this is done to prevent overfitting MDN-RNN to a specific 'z'.)
        latent = Normal(loc=states_[:, 0], scale=states_[:, 1])
        z_samples = latent.sample()

        states[:sequence_len] = z_samples[:-offset]
        next_states[:sequence_len] = z_samples[offset:]
        actions[:sequence_len] = actions_

        return [states, actions], [next_states]

    def __len__(self):
        return self.dataset.attrs["N_GAMES"]

    def close(self):
        self.dataset.close()


class MDN(nn.Module):
    def __init__(self, hidden_units, latent_dim, action_space, temperature, n_gaussians, num_layers=1):
        super(MDN, self).__init__()

        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.n_gaussians = n_gaussians
        self.num_layers = num_layers

        self.embedding = nn.Embedding.from_pretrained(torch.eye(action_space.num)) \
            if isinstance(action_space, hrl.environments.Discrete) else None
        self.lstm = nn.LSTM(input_size=(latent_dim + action_space.num),
                            hidden_size=hidden_units,
                            num_layers=num_layers,
                            batch_first=True)
        self.pi = nn.Linear(hidden_units, n_gaussians * latent_dim)
        self.mu = nn.Linear(hidden_units, n_gaussians * latent_dim)
        self.logsigma = nn.Linear(hidden_units, n_gaussians * latent_dim)

    def forward(self, latent, action):
        self.lstm.flatten_parameters()
        sequence_len = latent.size(1)
        if self.embedding:
            # Use one-hot representation for discrete actions
            x = torch.cat((latent, self.embedding(action).squeeze(dim=2)), dim=2)
        else:
            # Pass raw action vector for continuous actions
            x = torch.cat((latent, action.float()), dim=2)

        h, self.hidden = self.lstm(x, self.hidden)

        pi = self.pi(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim) / self.temperature
        pi = torch.softmax(pi, dim=2)

        logsigma = self.logsigma(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim)
        sigma = torch.exp(logsigma)

        mu = self.mu(h).view(-1, sequence_len, self.n_gaussians, self.latent_dim)

        return mu, sigma, pi

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device

        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_units, device=device)
        )


def build_rnn_model(rnn_params, latent_dim, action_space, model_path=None):
    """Builds MDN-RNN memory module, which model time dependencies.

    Args:
        rnn_params (dict): MDN-RNN parameters from .json config.
        latent_dim (int): Latent space dimensionality.
        action_space (hrl.environments.ActionSpace): Action space, discrete or continuous.
        model_path (str): Path to VAE ckpt. Taken from .json config if `None` (Default: None)

    Returns:
        TorchTrainer: Compiled MDN-RNN model wrapped in TorchTrainer, ready for training.
    """

    use_cuda = torch.cuda.is_available()

    def mdn_loss_function(pred, target):
        """Mixed Density Network loss function, see:
        https://mikedusenberry.com/mixture-density-networks"""

        mu, sigma, pi = pred

        sequence_len = mu.size(1)
        latent_dim = mu.size(3)
        target = target.view(-1, sequence_len, 1, latent_dim)

        loss = Normal(loc=mu, scale=sigma)
        loss = torch.exp(loss.log_prob(target))
        loss = torch.sum(loss * pi, dim=2)
        loss = -torch.log(loss + 1e-9)

        return torch.mean(loss)

    mdn = TorchTrainer(MDN(rnn_params['hidden_units'], latent_dim, action_space,
                           rnn_params['temperature'], rnn_params['n_gaussians']),
                       device_name='cuda' if use_cuda else 'cpu')

    mdn.compile(optimizer=optim.Adam(mdn.model.parameters(), lr=rnn_params['learning_rate']),
                loss=mdn_loss_function)

    model_path = get_model_path_if_exists(
        path=model_path, default_path=rnn_params['ckpt_path'], model_name="MDN-RNN")

    if model_path is not None:
        mdn.load_ckpt(model_path)
        log.info("Loaded MDN-RNN model weights from: %s", model_path)

    return mdn
