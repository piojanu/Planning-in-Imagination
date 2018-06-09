import os

from abc import ABCMeta, abstractmethod

from keras.optimizers import SGD


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
    def train(self, data, targets):
        """Perform training according to passed parameters in :method:`build` call.

        Args:
            data (numpy.Array): States to train on.
            targets (numpy.Array): Ground truth targets, depend on specific model.
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
    def load_checkpoint(self, folder, filename):
        """Loads parameters of the neural network from folder/filename.

        Args:
            folder (string): Directory for loading checkpoints from.
            filename (string): File name of saved nn checkpoint.
        """

        pass


class KerasNet(NeuralNet):
    """Artificial neural mind of planning."""

    def __init__(self, model, params={}):
        """Compile neural network model in Keras.

        Args:
            model (keras.Model): Neural network architecture.
            params (dict): Train/inference hyper-parameters. Available:
              * 'loss' (string)     : Loss function name, passed to model.compile(...) method.
                                      (Default: "MSE")
              * 'lr' (float)        : Learning rate of SGD with momentum optimizer. (Default: 0.01)
              * 'momentum' (float)  : Parameter that accelerates SGD in the relevant direction and
                                      dampens oscillations. (Default: 0.)
              * 'decay' (float)     : Learning rate decay over each update. (Default: 0.)
              * 'batch_size' (int)  : Training batch size. (Default: 32)
              * 'epochs' (int)      : Number of epochs to train the model. (Default: 1)
              * 'val_split' (float) : Fraction of the training data to be used as validation data.
                                      (Default: 0.)
        """

        model.compile(loss=params.get('loss', "MSE"),
                      optimizer=SGD(lr=params.get('lr', 0.01),
                                    momentum=params.get('momentum', 0.),
                                    decay=params.get('decay', 0.),
                                    nesterov=True))

        self.model = model
        self.batch_size = params.get('batch_size', 32)
        self.epochs = params.get('epochs', 1)
        self.val_split = params.get('val_split', 0.)

    def predict(self, state):
        """Do forward pass through nn, inference on state.

        Args:
            state (numpy.Array): State of game to inference on.

        Returns:
            numpy.Array: Inference result, depends on specific model.
        """

        return self.model.predict(state)[0]

    def train(self, data, targets):
        """Perform training according to passed parameters in `build` call.

        Args:
            data (numpy.Array): States to train on.
            targets (numpy.Array): Ground truth targets, depend on specific model.
        """

        self.model.fit(data, targets,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=self.val_split)

    def save_checkpoint(self, folder, filename):
        """Saves the current neural network (with its parameters) in folder/filename.

        Args:
            folder (string): Directory for storing checkpoints.
            filename (string): File name to save nn in, will have date/time appended.
        """

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        self.model.save_weights(filepath)

    def load_checkpoint(self, folder, filename):
        """Loads parameters of the neural network from folder/filename.

        Args:
            folder (string): Directory for loading checkpoints from.
            filename (string): File name of saved nn checkpoint.
        """

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.model.load_weights(filepath)
