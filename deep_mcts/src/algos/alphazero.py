import logging as log
import numpy as np

from .value_function import Planner as VFPlanner
from tree.basic import Edge

from keras.backend import image_data_format
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, Reshape
from keras.models import Model
from keras.regularizers import l2


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
    """

    l2_reg = params.get("l2_regularizer", 0.001)
    dropout = params.get("dropout", 0.4)

    DATA_FORMAT = image_data_format()
    BOARD_HEIGHT, BOARD_WIDTH = game.get_board_size()
    ACTION_SIZE = game.get_action_size()

    def conv2d_n_batchnorm(x, filters, padding='same'):
        conv = Conv2D(filters, kernel_size=(3, 3), padding=padding,
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
    x = conv2d_n_batchnorm(x, filters=256, padding='valid')
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)

    pi = Dense(ACTION_SIZE, activation='softmax', kernel_regularizer=l2(l2_reg), name='pi')(x)
    value = Dense(1, activation='tanh', kernel_regularizer=l2(l2_reg), name='value')(x)

    return Model(inputs=boards_input, outputs=[pi, value])


class Planner(VFPlanner):
    """AlphaZero search operations and planning logic."""

    def __init__(self, model, nn, params={}):
        """Initialize MCTS object

        Args:
            model (Game): Perfect information dynamics/game.
            nn (NeuralNet): Artificial neural mind used to evaluate leaf states.
            params (dict): MCTS search hyper-parameters. Available:
              * 'c' (float)           : UCT exploration-exploitation trade-off param (Default: 1.)
              * 'gamma' (float)       : Discounting factor for value. (Default: 1./no discounting)
              * 'n_simulations' (int) : Number of simulations to perform before choosing action.
                                        (Default: 25)
        """

        VFPlanner.__init__(self, model, nn, params)

    def evaluate(self, leaf_node):
        """Expand and evaluate leaf node.

        Args:
            leaf_node (object): Leaf node to expand and evaluate.

        Returns:
            (float): Node (state) value.
        """

        # Evaluate state
        pi, value = self.nn.predict(np.expand_dims(leaf_node.state, axis=0))

        # Take first element in batch
        pi = pi[0]
        value = value[0]

        # Get valid actions
        valid_moves = self.model.get_valid_moves(leaf_node.state, leaf_node.player)

        # Renormalize valid actions probabilities
        probs = np.zeros_like(pi)
        probs[valid_moves] = pi[valid_moves]

        sum_probs = np.sum(probs)
        if sum_probs <= 0:
            # If all valid moves were masked make all valid moves equally probable
            log.warn("All valid moves were masked, do workaround!")
            probs[valid_moves] = 1
        probs = probs / sum_probs

        # Expand node with edges
        edges = {}
        for m in valid_moves:
            edges[m] = Edge(prior=probs[m])

        leaf_node.expand(edges)

        # Get value from result array
        return value[0]
