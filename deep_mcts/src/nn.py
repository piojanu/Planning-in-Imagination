import logging as log
import os

from abc import ABCMeta, abstractmethod

from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping
from keras.backend import image_data_format
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, Reshape
from keras.models import Model
from keras.regularizers import l2


class NeuralNet(metaclass=ABCMeta):
    """Artificial neural mind of planning."""

    @abstractmethod
    def __init__(self, arch, params={}):
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
            state (numpy.Array): State of game to inference on.

        Returns:
            numpy.Array: Inference result, depends on specific model.
        """

        pass

    @abstractmethod
    def train(self, data, targets, callbacks=[]):
        """Perform training according to passed parameters in :method:`build` call.

        Args:
            data (numpy.Array): States to train on.
            targets (numpy.Array): Ground truth targets, depend on specific model.
            callbacks (list): Extra callbacks to pass to keras model fit method. (Default: [])
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

    def __init__(self, model, params={}):
        """Compile neural network model in Keras.

        Args:
            model (keras.Model): Neural network architecture.
            params (dict): Train/inference hyper-parameters. Available:
              * 'batch_size' (int)  : Training batch size. (Default: 32)
              * 'epochs' (int)      : Number of epochs to train the model. (Default: 50)
              * 'val_split' (float) : Fraction of the training data to be used as validation data.
                                      Float >= 0 and < 1.. (Default: 0.2)
              * 'patience'          : Number of epochs with no improvement in validation loss after
                                      which training will be stopped. You need to set val_split > 0
                                      in order to have it work. Set to -1 for no early stopping.
                                      (Default: 5)
              * 'save_training_log_path' (string) : where to save nn train logs.
                                                    (Default: "./logs/training.log")
        """

        self.model = model
        self.batch_size = params.get('batch_size', 32)
        self.epochs = params.get('epochs', 50)
        self.val_split = params.get('val_split', 0.2)
        self.patience = params.get('patience', 5)

        # Initialize EarlyStopping callback if validation set is present
        self.callbacks = []
        if self.patience > 0:
            if self.val_split > 0:
                self.callbacks.append(EarlyStopping(monitor='val_loss',
                                                    patience=self.patience))
            else:
                log.warn("Early stopping DISABLED! Patience > 0 byt val_split <= 0.")

        # Add CSVLogger
        self.callbacks.append(CSVLogger(
            params.get('save_training_log_path', './logs/training.log'), append=True))

    def predict(self, state):
        """Do forward pass through nn, inference on state.

        Args:
            state (numpy.Array): State of game to inference on.

        Returns:
            numpy.Array: Inference result, depends on specific model.
        """

        return self.model.predict(state)

    def train(self, data, targets, callbacks):
        """Perform training according to passed parameters in `build` call.

        Args:
            data (numpy.Array): States to train on.
            targets (numpy.Array): Ground truth targets, depend on specific model.
            callbacks (list): Extra callbacks to pass to keras model fit method. (Default: [])
        """

        self.model.fit(data, targets,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=self.val_split,
                       callbacks=self.callbacks + callbacks)

    def save_checkpoint(self, folder, filename):
        """Saves the current neural network (with its parameters) in folder/filename.

        Args:
            folder (string): Directory for storing checkpoints.
            filename (string): File name to save nn in, will have date/time appended.
        """

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            log.warn("Checkpoint directory does not exist! Creating directory {}".format(folder))
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
          * 'l2_regularizer' (float) : L2 weight decay rate.
                                       (Default: 0.001)
          * 'dropout' (float)        : Dense layers dropout rate.
                                       (Default: 0.4)
          * 'loss' (string)     : Loss function name, passed to keras model.compile(...) method.
                                  (Default: "MSE")
          * 'lr' (float)        : Learning rate of SGD with momentum optimizer. Float > 0.
                                  (Default: 0.01)
          * 'momentum' (float)  : Parameter that accelerates SGD in the relevant direction and
                                  dampens oscillations. Float >= 0 (Default: 0.)
          * 'decay' (float)     : Learning rate decay over each update. Float >= 0. (Default: 0.)
    """

    decay = params.get('decay', 0.)
    dropout = params.get("dropout", 0.4)
    l2_reg = params.get("l2_regularizer", 0.001)
    loss = params.get('loss', "MSE")
    lr = params.get('lr', 0.01)
    momentum = params.get('momentum', 0.)

    DATA_FORMAT = image_data_format()
    BOARD_HEIGHT, BOARD_WIDTH = game.get_board_size()
    ACTION_SIZE = game.get_action_size()

    def conv2d_n_batchnorm(x, filters, padding='same'):
        conv = Conv2D(filters, kernel_size=(3, 3), padding=padding, strides=2,
                      kernel_regularizer=l2(l2_reg))(x)
        if DATA_FORMAT == 'channels_first':
            bn = BatchNormalization(axis=1)(conv)
        else:
            bn = BatchNormalization(axis=3)(conv)
        return Activation(activation='relu')(bn)

    # Build model

    boards_input = Input(shape=(BOARD_HEIGHT, BOARD_WIDTH))

    if DATA_FORMAT == 'channels_first':
        x = Reshape((1, BOARD_HEIGHT, BOARD_WIDTH))(boards_input)
    else:
        x = Reshape((BOARD_HEIGHT, BOARD_WIDTH, 1))(boards_input)

    x = conv2d_n_batchnorm(x, filters=128, padding='same')
    x = conv2d_n_batchnorm(x, filters=256, padding='same')
    x = conv2d_n_batchnorm(x, filters=512, padding='same')
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)

    pi = Dense(ACTION_SIZE, activation='softmax', kernel_regularizer=l2(l2_reg), name='pi')(x)
    value = Dense(1, activation='tanh', kernel_regularizer=l2(l2_reg), name='value')(x)

    model = Model(inputs=boards_input, outputs=[pi, value])

    model.compile(loss=loss,
                  optimizer=SGD(lr=lr,
                                momentum=momentum,
                                decay=decay,
                                nesterov=True),
                  metrics=['accuracy'])
    return model
