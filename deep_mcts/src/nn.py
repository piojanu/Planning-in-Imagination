import logging as log
import os

from abc import ABCMeta, abstractmethod

from keras.optimizers import SGD
from keras.callbacks import CSVLogger, EarlyStopping


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
              * 'loss' (string)     : Loss function name, passed to keras model.compile(...) method.
                                      (Default: "MSE")
              * 'lr' (float)        : Learning rate of SGD with momentum optimizer. Float > 0.
                                      (Default: 0.01)
              * 'momentum' (float)  : Parameter that accelerates SGD in the relevant direction and
                                      dampens oscillations. Float >= 0 (Default: 0.)
              * 'decay' (float)     : Learning rate decay over each update. Float >= 0. (Default: 0.)
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
        self.loss = params.get('loss', "MSE")
        self.lr = params.get('lr', 0.01)
        self.momentum = params.get('momentum', 0.)
        self.decay = params.get('decay', 0.)
        self.batch_size = params.get('batch_size', 32)
        self.epochs = params.get('epochs', 50)
        self.val_split = params.get('val_split', 0.2)
        self.patience = params.get('patience', 5)

        # Initialize EarlyStopping callback if validation set is present
        self.callbacks = []
        if self.patience > 0:
            if self.val_split > 0:
                self.callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience))
            else:
                log.warn("Early stopping DISABLED! Patience > 0 byt val_split <= 0.")

        # Add CSVLogger
        self.callbacks.append(CSVLogger(
            params.get('save_training_log_path', './logs/training.log'), append=True))

        model.compile(loss=self.loss,
                      optimizer=SGD(lr=self.lr,
                                    momentum=self.momentum,
                                    decay=self.decay,
                                    nesterov=True),
                      metrics=['accuracy'])

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
