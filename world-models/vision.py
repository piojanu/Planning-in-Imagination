import logging as log
import random

import keras.backend as K
import numpy as np
import os.path
import h5py

from humblerl import Vision, Callback
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam

from utils import get_model_path_if_exists


class StoreVaeTransitions(Callback):
    """Save transitions to HDF5 file in one dataset:
        * 'states': Keeps transition's state (e.g. image).
        Datasets are organized in such a way, that you can access transition 'I' by accessing
        'I'-th position in all three datasets.

        HDF5 file also keeps meta-informations (attributes) as such:
        * 'N_TRANSITIONS': Datasets size (number of transitions).
        * 'N_GAMES': From how many games those transitions come from.
        * 'CHUNK_SIZE': HDF5 file chunk size (batch size should be multiple of it).
        * 'STATE_SHAPE': Shape of environment's state ('(next_)states' dataset element shape).
    """

    def __init__(self, state_shape, out_path, shuffle=True, min_transitions=10000, chunk_size=128, dtype=np.uint8):
        """Save transitions to HDF5 file.

        Args:
            state_shape (tuple): Shape of environment's state.
            out_path (str): Path to HDF5 file where transition will be stored.
            shuffle (bool): If data should be shuffled (in subsets of `min_transitions` number of
                transitions). (Default: True)
            min_transitions (int): Minimum size of dataset in transitions number. Also, whenever
                this amount of transitions is gathered, data is shuffled (if requested) and stored
                on disk. (Default: 10000)
            chunk_size (int): Chunk size in transitions. For efficiency reasons, data is saved
                to file in chunks to limit the disk usage (chunk is smallest unit that get fetched
                from disk). For best performance set it to training batch size and in e.g. Keras
                use shuffle='batch'/False. Never use shuffle=True, as random access to hdf5 is
                slow. (Default: 128)
            dtype (np.dtype): Data type of states. (Default: np.uint8)
        """

        self.out_path = out_path
        self.dataset_size = min_transitions
        self.shuffle_chunk = shuffle
        self.min_transitions = min_transitions
        self.state_dtype = dtype

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

        # Create dataset for states
        # NOTE: We save states as np.uint8 dtype because those are RGB pixel values.
        self.out_states = self.out_file.create_dataset(
            name="states", dtype=dtype, chunks=(chunk_size, *state_shape),
            shape=(self.dataset_size, *state_shape), maxshape=(None, *state_shape),
            compression="lzf")

        self.transition_count = 0
        self.game_count = 0

        self.states = []

    def on_step_taken(self, step, transition, info):
        self.states.append(transition.state)

        if transition.is_terminal:
            self.game_count += 1

        self.transition_count += 1
        if self.transition_count % self.min_transitions == 0:
            self._save_chunk()

    def on_loop_end(self, is_aborted):
        if len(self.states) > 0:
            self._save_chunk()

        # Close file
        self.out_file.close()

    def _save_chunk(self):
        """Save `states`  to HDF5 file. Clear the buffers.
        Update transition and games count in HDF5 file."""

        # Resize datasets if needed
        if self.transition_count > self.dataset_size:
            self.out_states.resize(self.transition_count, axis=0)
            self.dataset_size = self.transition_count

        n_transitions = len(self.states)
        start = self.transition_count - n_transitions

        assert n_transitions > 0, "Nothing to save!"

        if self.shuffle_chunk:
            states = []

            for idx in random.sample(range(n_transitions), n_transitions):
                states.append(self.states[idx])
        else:
            states = self.states

        self.out_states[start:self.transition_count] = \
            np.array(states, dtype=self.state_dtype)

        self.out_file.attrs["N_TRANSITIONS"] = self.transition_count
        self.out_file.attrs["N_GAMES"] = self.game_count

        self.states = []


class VAEVision(Vision):
    def __init__(self, model, state_processor_fn):
        """Initialize vision processors.

        Args:
            model (keras.Model): Keras VAE encoder.
            state_processor_fn (function): Function for state processing. It should
                take raw environment state as an input and return processed state.
        """

        # NOTE: [0:2] <- it gets latent space mean (mu) and logvar, then concatenate batch dimension
        #       (batch size is one, after concatenate we get array '2 x latent space dim').
        super(VAEVision, self).__init__(lambda state: np.concatenate(
            model.predict(state_processor_fn(state)[np.newaxis, :] / 255.)[0:2]))


def build_vae_model(vae_params, input_shape, model_path=None):
    """Builds VAE encoder, decoder using Keras Model and VAE loss.

    Args:
        vae_params (dict): VAE parameters from .json config.
        input_shape (tuple): Input to encoder shape (state shape).
        model_path (str): Path to VAE ckpt. Taken from .json config if `None` (Default: None)

    Returns:
        keras.models.Model: Compiled VAE, ready for training.
        keras.models.Model: Encoder.
        keras.models.Model: Decoder.
    """

    if K.image_data_format() == 'channel_first':
        raise ValueError("Channel first backends aren't supported!")

    # Encoder img -> mu, logvar #

    encoder_input = Input(shape=input_shape)

    h = Conv2D(32, activation='relu', kernel_size=4, strides=2)(encoder_input)  # -> 31x31x32
    h = Conv2D(64, activation='relu', kernel_size=4, strides=2)(h)              # -> 14x14x64
    h = Conv2D(128, activation='relu', kernel_size=4, strides=2)(h)             # -> 6x6x128
    h = Conv2D(256, activation='relu', kernel_size=4, strides=2)(h)             # -> 2x2x256

    batch_size = K.shape(h)[0]  # Needed to sample latent vector
    h_shape = K.int_shape(h)    # Needed to reconstruct in decoder

    h = Flatten()(h)
    mu = Dense(vae_params['latent_space_dim'])(h)
    logvar = Dense(vae_params['latent_space_dim'])(h)

    # Sample latent vector #

    def sample(args):
        mu, logvar = args
        return mu + K.exp(logvar) * K.random_normal(
            shape=(batch_size, vae_params['latent_space_dim']))

    z = Lambda(sample, output_shape=(vae_params['latent_space_dim'],))([mu, logvar])

    encoder = Model(encoder_input, [mu, logvar, z], name='Encoder')
    encoder.summary(print_fn=lambda x: log.debug('%s', x))

    # Decoder z -> img #

    decoder_input = Input(shape=(vae_params['latent_space_dim'],))

    h = Reshape(h_shape[1:])(
        Dense(h_shape[1] * h_shape[2] * h_shape[3], activation='relu')(decoder_input)
    )

    h = Conv2DTranspose(128, activation='relu', kernel_size=4, strides=2)(h)     # -> 6x6x128
    h = Conv2DTranspose(64, activation='relu', kernel_size=4, strides=2)(h)      # -> 14x14x64
    h = Conv2DTranspose(32, activation='relu', kernel_size=4, strides=2)(h)      # -> 30x30x32
    out = Conv2DTranspose(3, activation='sigmoid', kernel_size=6, strides=2)(h)  # -> 64x64x3

    decoder = Model(decoder_input, out, name='Decoder')
    decoder.summary(print_fn=lambda x: log.debug('%s', x))

    # VAE loss #

    def elbo_loss(target, pred):
        # NOTE: You use K.reshape to preserve batch dim. K.flatten doesn't work like flatten layer
        #       and flatten batch dim. too!
        # NOTE 2: K.binary_crossentropy does element-wise crossentropy as you need (it calls
        #         tf.nn.sigmoid_cross_entropy_with_logits in backend), but Keras loss
        #         binary_crossentropy would average over spatial dim. You sum it as you don't want
        #         to weight reconstruction loss lower (divide by H * W * C) then KL loss.
        reconstruction_loss = K.sum(
            K.binary_crossentropy(
                K.reshape(target, [batch_size, -1]), K.reshape(pred, [batch_size, -1])
            ),
            axis=1
        )

        # NOTE: Closed form of KL divergence for Gaussians.
        #       See Appendix B from VAE paper (Kingma 2014):
        #       https://arxiv.org/abs/1312.6114
        KL_loss = K.sum(
            1. + logvar - K.square(mu) - K.exp(logvar),
            axis=1
        ) / 2

        return reconstruction_loss - KL_loss

    # Build and compile VAE model #

    decoder_output = decoder(encoder(encoder_input)[2])
    vae = Model(encoder_input, decoder_output, name='VAE')
    vae.compile(optimizer=Adam(lr=vae_params['learning_rate']), loss=elbo_loss)
    vae.summary(print_fn=lambda x: log.debug('%s', x))

    model_path = get_model_path_if_exists(
        path=model_path, default_path=vae_params['ckpt_path'], model_name="VAE")

    if model_path is not None:
        vae.load_weights(model_path)
        log.info("Loaded VAE model weights from: %s", model_path)

    return vae, encoder, decoder
