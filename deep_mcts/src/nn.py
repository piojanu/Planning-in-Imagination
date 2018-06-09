from abc import ABCMeta, abstractmethod


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

    def __init__(self, arch, params={}):
        """Compile neural network model in Keras.

        Args:
            arch (keras.Model): Neural network architecture.
            params (dict): Train/inference hyper-parameters. Available:
              * '...' (...) : ...
        """

        # TODO (pj): Implement compilation on Keras model.
        # raise NotImplementedError()

    def predict(self, state):
        """Do forward pass through nn, inference on state.

        Args:
            state (numpy.Array): State of game to inference on.

        Returns:
            numpy.Array: Inference result, depends on specific model.
        """

        # TODO (pj): Implement inference on Keras model.
        raise NotImplementedError()

    def train(self, data, targets):
        """Perform training according to passed parameters in `build` call.

        Args:
            data (numpy.Array): States to train on.
            targets (numpy.Array): Ground truth targets, depend on specific model.
        """

        # TODO (pj): Implement training of Keras model on given data.
        raise NotImplementedError()

    def save_checkpoint(self, folder, filename):
        """Saves the current neural network (with its parameters) in folder/filename.

        Args:
            folder (string): Directory for storing checkpoints.
            filename (string): File name to save nn in, will have date/time appended.
        """

        # TODO (pj): Implement saving Keras model.
        raise NotImplementedError()

    def load_checkpoint(self, folder, filename):
        """Loads parameters of the neural network from folder/filename.

        Args:
            folder (string): Directory for loading checkpoints from.
            filename (string): File name of saved nn checkpoint.
        """

        # TODO (pj): Implement loading Keras model.
        raise NotImplementedError()
