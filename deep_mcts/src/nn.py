from abc import ABCMeta, abstractmethod


class KerasNet(metaclass=ABCMeta):
    """Artificial neural mind of planning."""

    @abstractmethod
    def build(self, **kwargs):
        """Build neural network model in Keras.

        Args:
            **kwargs: Hyper-parameters, depend on specific model implementation.
        """

        pass

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
