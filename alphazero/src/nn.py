import logging as log
import os

from abc import ABCMeta, abstractmethod

from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping
from keras.backend import image_data_format
from keras.layers import Activation, add, BatchNormalization, Conv2D, Dense, Flatten, GlobalAveragePooling2D, Input, Reshape
from keras.models import Model
from keras.regularizers import l2


class NeuralNet(metaclass=ABCMeta):
    """Artificial neural mind of planning."""

    @abstractmethod
    def __init__(self, arch, params):
        """Compile neural network model.

        Args:
            arch (object): Neural network architecture.
            params (dict): Train/inference hyper-parameters.
        """

        pass

    @abstractmethod
    def predict(self, state):
        """Do forward pass through nn, inference on state.

        Args:
            state (np.ndarray): State of game to inference on.

        Returns:
            np.ndarray: Inference result, depends on specific model.
        """

        pass

    @abstractmethod
    def train(self, data, targets, callbacks=None):
        """Perform training according to passed parameters in :method:`build` call.

        Args:
            data (np.ndarray): States to train on.
            targets (np.ndarray): Ground truth targets, depend on specific model.
            callbacks (list): Extra callbacks to pass to keras model fit method. (Default: None)
         """

        pass

    @abstractmethod
    def save_checkpoint(self, folder, filename):
        """Saves the current neural network (with its parameters) in folder/filename.

        Args:
            folder (string): Directory for storing checkpoints.
            filename (string): File name to save nn in, will have date/time appended.
        """

        pass

    @abstractmethod
    def load_checkpoint(self, path, filename=None):
        """Loads parameters of the neural network from folder/filename.

        Args:
            path (string): Directory for loading checkpoints from or full path to file
                           if filename is None.
            filename (string): File name of saved nn checkpoint. (Default: None)
        """

        pass


class KerasNet(NeuralNet):
    """Artificial neural mind of planning."""

    def __init__(self, model, params):
        """Compile neural network model in Keras.

        Args:
            model (keras.Model): Neural network architecture.
            params (dict): Train/inference hyper-parameters. Available:
              * 'batch_size' (int)  : Training batch size. (Default: 32)
              * 'epochs' (int)      : Number of epochs to train the model. (Default: 50)
              * 'save_training_log_path' (string) : where to save nn train logs.
                                                    (Default: "./logs/training.log")
        """

        self.model = model
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']

        # Initialize callbacks list with CSVLogger
        self.callbacks = [
            CSVLogger(params.get('save_training_log_path', './logs/training.log'), append=True)]

    def predict(self, state):
        """Do forward pass through nn, inference on state.

        Args:
            state (np.ndarray): State of game to inference on.

        Returns:
            np.ndarray: Inference result, depends on specific model.
        """

        return self.model.predict(state)

    def train(self, data, targets, initial_epoch=0, callbacks=None):
        """Perform training according to passed parameters in `build` call.

        Args:
            data (np.ndarray): States to train on.
            targets (np.ndarray): Ground truth targets, depend on specific model.
            initial_epoch (int): Epoch at which to start training. (Default: 0)
            callbacks (list): Extra callbacks to pass to keras model fit method. (Default: None)

        Return:
            int: Number of training epochs to this moment.
        """

        epochs = self.epochs + initial_epoch

        if callbacks is None:
            callbacks = []

        self.model.fit(data, targets,
                       batch_size=self.batch_size,
                       epochs=epochs,
                       initial_epoch=initial_epoch,
                       callbacks=self.callbacks + callbacks)

        return epochs

    def save_checkpoint(self, folder, filename):
        """Saves the current neural network (with its parameters) in folder/filename.

        Args:
            folder (string): Directory for storing checkpoints.
            filename (string): File name to save nn in, will have date/time appended.
        """

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            log.warning("Checkpoint does directory not exist! Creating directory %s", folder)
            os.mkdir(folder)

        self.model.save_weights(filepath)

    def load_checkpoint(self, path, filename=None):
        """Loads parameters of the neural network from folder/filename.

        Args:
            path (string): Directory for loading checkpoints from or full path to file
                           if filename is None.
            filename (string): File name of saved nn checkpoint. (Default: None)
        """

        if filename is None:
            filepath = path
        else:
            filepath = os.path.join(path, filename)

        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.model.load_weights(filepath)


def build_keras_nn(game, params):
    """Build neural network model in Keras.

    Args:
        game (Game): Perfect information dynamics/game. Used to get information
                     like action/state space sizes etc.
        params (dict): Neural Net architecture hyper-parameters. Available:
          * architecture related      : ResNet like architecture description. If conv_filters = -1 or
                                        residual_filters = -1, then no conv layer/residual blocks.
          * 'feature_extractor' (str) : "agz" (Default) - heads are the same like in AlphaGo Zero,
                                        "avgpool"       - global avgpool after residual tower,
                                        "flatten"       - flatten residual tower output.
                                        After 'avgpool' and 'flatten' there is FC layer controlled
                                        with 'dense_size' parameter.
          * 'loss' (string)           : Loss function name, passed to keras model.compile(...) method.
                                        (Default: ["categorical_crossentropy", "mean_squared_error"])
          * 'l2_regularizer' (float)  : L2 weight decay rate.
                                        (Default: 0.0001)
          * 'lr' (float)              : Learning rate of SGD with momentum optimizer. Float > 0.
                                        (Default: 0.1)
          * 'momentum' (float)        : Parameter that accelerates SGD in the relevant direction and
                                        dampens oscillations. Float >= 0 (Default: 0.9)
    """

    conv_filters = params["conv_filters"]
    conv_kernel = params["conv_kernel"]
    conv_stride = params["conv_stride"]
    residual_bottleneck = params["residual_bottleneck"]
    residual_filters = params["residual_filters"]
    residual_kernel = params["residual_kernel"]
    residual_num = params["residual_num"]
    feature_extractor = params["feature_extractor"]
    dense_size = params["dense_size"]

    loss = params['loss']
    l2_reg = params["l2_regularizer"]
    lr = params['lr']
    momentum = params['momentum']

    DATA_FORMAT = image_data_format()
    BOARD_HEIGHT, BOARD_WIDTH = game.getBoardSize()
    ACTION_SIZE = game.getActionSize()

    def conv2d_n_batchnorm(x, filters, kernel_size, strides=1, shortcut=None):
        conv = Conv2D(filters, kernel_size=kernel_size, strides=strides,
                      padding="same", kernel_regularizer=l2(l2_reg), data_format=DATA_FORMAT)(x)

        if DATA_FORMAT == 'channels_first':
            bn = BatchNormalization(axis=1)(conv)
        else:
            bn = BatchNormalization(axis=3)(conv)

        if shortcut is not None:
            out = add([bn, shortcut])
        else:
            out = bn

        return Activation(activation='relu')(out)

    def residual_block(x, filters, bottleneck, kernel_size):
        y = conv2d_n_batchnorm(x, bottleneck, kernel_size=1, strides=1)
        y = conv2d_n_batchnorm(y, bottleneck, kernel_size, strides=1)
        return conv2d_n_batchnorm(y, filters, kernel_size=1, strides=1, shortcut=x)

    # Add batch dimension to inputs
    boards_input = Input(shape=(BOARD_HEIGHT, BOARD_WIDTH))
    if DATA_FORMAT == 'channels_first':
        x = Reshape((1, BOARD_HEIGHT, BOARD_WIDTH))(boards_input)
    else:
        x = Reshape((BOARD_HEIGHT, BOARD_WIDTH, 1))(boards_input)

    # Input convolution
    if conv_filters > 0:
        x = conv2d_n_batchnorm(
            x, filters=conv_filters, kernel_size=conv_kernel, strides=conv_stride)

    # Tower of residual blocks
    if residual_filters > 0:
        if conv_filters != residual_filters:
            # Add additional layer to even out the number of filters between input CNN
            # and residual blocks, so that residual shortcut connection works properly
            x = conv2d_n_batchnorm(x, filters=residual_filters, kernel_size=residual_kernel,
                                   strides=1)
        for _ in range(residual_num):
            x = residual_block(x, residual_filters, residual_bottleneck, residual_kernel)

    # Final feature extractors
    if feature_extractor == "agz":
        pi = Flatten()(conv2d_n_batchnorm(x, filters=2, kernel_size=1, strides=1))
        value = Flatten()(conv2d_n_batchnorm(x, filters=1, kernel_size=1, strides=1))
        value = Dense(dense_size, activation='relu',
                      kernel_regularizer=l2(l2_reg))(value)
    elif feature_extractor == "avgpool":
        x = GlobalAveragePooling2D(data_format=DATA_FORMAT)(x)
        pi = value = Dense(dense_size, activation='relu',
                           kernel_regularizer=l2(l2_reg))(x)
    elif feature_extractor == "flatten":
        x = Flatten()(x)
        pi = value = Dense(dense_size, activation='relu',
                           kernel_regularizer=l2(l2_reg))(x)
    else:
        raise ValueError("Unknown feature extractor! Possible values: 'agz', 'avgpool', 'flatten'")

    # Heads
    pi = Dense(ACTION_SIZE, activation='softmax',
               kernel_regularizer=l2(l2_reg), name='pi')(pi)
    value = Dense(1, activation='tanh', kernel_regularizer=l2(
        l2_reg), name='value')(value)

    # Create model
    model = Model(inputs=boards_input, outputs=[pi, value])

    # Compile model
    model.compile(loss=loss,
                  optimizer=SGD(lr=lr,
                                momentum=momentum,
                                nesterov=True),
                  metrics=['accuracy'])

    # Log model architecture
    model.summary(print_fn=lambda x: log.debug("%s", x))
    return model
