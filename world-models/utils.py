import json
import h5py
import numpy as np
import os
import logging as log
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
from keras.utils import Sequence
from skimage.transform import resize
from tqdm import tqdm

import humblerl as hrl


class Config(object):
    def __init__(self, config_path, is_debug, allow_render):
        """Loads custom configuration, unspecified parameters are taken from default configuration.

        Args:
            config_path (str): Path to .json file with custom configuration
            is_debug (bool): Specify to enable debugging features
            allow_render (bool): Specify to enable render/plot features
        """
        with open(os.path.join(os.path.dirname(__file__), "config.json.dist")) as config_file:
            default_config = json.loads(config_file.read())

        if os.path.exists(config_path):
            with open(config_path) as custom_config_file:
                custom_config = json.loads(custom_config_file.read())
        else:
            custom_config = {}

        # Merging default and custom configs, for repeating keys second dict overwrites values
        self.general = {**default_config["general"], **custom_config.get("general", {})}
        self.es = {**default_config["es_training"], **custom_config.get("es_training", {})}
        self.rnn = {**default_config["rnn_training"], **custom_config.get("rnn_training", {})}
        self.vae = {**default_config["vae_training"], **custom_config.get("vae_training", {})}
        self.is_debug = is_debug
        self.allow_render = allow_render


class StoreTransitions(hrl.Callback):
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

    def __init__(self, out_path, state_shape, action_space, min_transitions=10000, min_episodes=1000, chunk_size=128,
                 state_dtype=np.uint8):
        """Initialize memory data storage.

        Args:
            out_path (str): Path to output hdf5 file.
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
        self.state_shape = state_shape
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
        self.out_file.attrs["STATE_SHAPE"] = state_shape
        self.out_file.attrs["ACTION_DIM"] = self.action_dim

        # Create datasets
        self.out_states = self.out_file.create_dataset(
            name="states", dtype=state_dtype, chunks=(chunk_size, *state_shape),
            shape=(self.dataset_size, *state_shape), maxshape=(None, *state_shape),
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


class HDF5DataGenerator(Sequence):
    """Generates data for Keras model from bug HDF5 files."""

    def __init__(self, hdf5_path, dataset_X, dataset_y, batch_size,
                 start=None, end=None, preprocess_fn=None):
        """Initialize data generator.

        Args:
            hdf5_path (str): Path to HDF5 file with data.
            dataset_X (str): Dataset's name with data.
            dataset_y (str): Dataset's name with targets.
            batch_size (int): Size of batch to return.
            start (int): Index where to start (inclusive) reading data/targets from dataset.
                If `None`, then it starts from the beginning. (Default: None)
            end (int): Index where to end (exclusive) reading data/targets from dataset.
                If `None`, then it reads dataset to the end. (Default: None)
            preprocess_fn (func): Function which accepts two arguments (batch of data and targets).
                It should return preprocessed batch (two values, data and targets!). If `None`, then
                no preprocessing is done. (Default: None)
        """

        hfile = h5py.File(hdf5_path, 'r')
        self.X = hfile[dataset_X]
        self.y = hfile[dataset_y]
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn

        if start is None:
            self.start = 0
        else:
            self.start = start

        if end is None:
            self.end = len(self.X)
        else:
            self.end = end

    def __len__(self):
        """Denotes the number of batches per epoch.

        Return:
            int: Number of batches in epoch.
        """

        return int(np.ceil((self.end - self.start) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data.

        Args:
            idx (int): Batch index.

        Return:
            np.ndarray: Batch of training examples.
            np.ndarray: Batch of targets.
        """

        start = self.start + idx * self.batch_size
        end = min(start + self.batch_size, self.end)

        X = self.X[start:end]
        y = self.y[start:end]

        if self.preprocess_fn is not None:
            X, y = self.preprocess_fn(X, y)

        return X, y


class TqdmStream(object):
    @classmethod
    def write(_, msg):
        tqdm.write(msg, end='')

    @classmethod
    def flush(_):
        pass


def state_processor(img, state_shape, crop_range):
    """Resize states to `state_shape` with cropping of `crop_range`.

    Args:
        img (np.ndarray): Image to crop and resize.
        state_shape (tuple): Output shape. Default: [64, 64, 3]
        crop_range (string): Range to crop as indices of array. Default: "[30:183, 28:131, :]"
    Return:
        np.ndarray: Cropped and reshaped to `state_shape` image.
    """

    # Crop image to `crop_range`, removes e.g. score bar
    img = eval("img" + crop_range)

    # Resize to 64x64 and cast to 0..255 values if requested
    return resize(img, state_shape, mode='constant') * 255


def get_model_path_if_exists(path, default_path, model_name):
    """Resize states to `state_shape` with cropping of `crop_range`.

    Args:
        path (string): Specified path to model
        default_path (string): Specified path to model
        model_name (string): Model name ie. VAE

    Returns:
        Path to model or None, depends whether first or second path exist
    """
    if path is None:
        if os.path.exists(default_path):
            path = default_path
        else:
            log.info("%s weights in \"%s\" doesn't exist! Starting tabula rasa.", model_name, path)
    elif not os.path.exists(path):
        raise ValueError("{} weights in \"{}\" path doesn't exist!".format(model_name, path))
    return path


def limit_gpu_memory_usage():
    """This function makes that we don't allocate more graphics memory than we need.
       For TensorFlow, we need to set `alow_growth` flag to True.
       For PyTorch, this is the default behavior.

    """

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))


def create_directory(dirname):
    """Create directory recursively, if it doesn't exit

    Args:
        dirname (str): Name of directory (path, e.g. "path/to/dir/")
    """
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)


def force_cpu():
    """Force using CPU"""

    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def mute_tf_logs_if_needed():
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
