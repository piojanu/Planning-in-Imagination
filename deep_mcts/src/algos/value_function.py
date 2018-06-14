import numpy as np

from mcts import MCTS
from nn import KerasNet
from tree.basic import Node, Edge

from keras.backend import image_data_format
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Reshape
from keras.models import Sequential
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

    def conv2d_n_batchnorm(model, filters, padding='same'):
        model.add(Conv2D(filters, kernel_size=(3, 3), padding=padding,
                         kernel_regularizer=l2(l2_reg)))
        if DATA_FORMAT == 'channels_first':
            model.add(BatchNormalization(axis=1))
        else:
            model.add(BatchNormalization(axis=3))
        model.add(Activation(activation='relu'))

    model = Sequential()
    if DATA_FORMAT == 'channels_first':
        model.add(Reshape((1, BOARD_HEIGHT, BOARD_WIDTH), input_shape=(BOARD_HEIGHT, BOARD_WIDTH)))
    else:
        model.add(Reshape((BOARD_HEIGHT, BOARD_WIDTH, 1), input_shape=(BOARD_HEIGHT, BOARD_WIDTH)))
    conv2d_n_batchnorm(model, filters=128, padding='same')
    conv2d_n_batchnorm(model, filters=256, padding='valid')
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='tanh', kernel_regularizer=l2(l2_reg)))

    return model


class Planner(MCTS):
    """MCTS search operations and planning logic."""

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

        MCTS.__init__(self, model, nn, params)

        self.c = params.get('c', 1.)
        self.gamma = params.get('gamma', 1.)

    def simulate(self, start_node):
        """Search through tree from start node to leaf.

        Args:
            start_node (Node): Where to start the search.

        Returns:
            (Node): Leaf node.
            (list): List of edges that make path from start node to leaf node.
        """

        current_node = start_node
        path = []

        while True:
            action_edge = current_node.select_edge(self.c)
            if action_edge is None:
                # This is leaf node, return now
                return current_node, path

            path.append(action_edge[1])
            next_node = action_edge[1].next_node

            if next_node is None:
                # This edge wasn't explored yet, create leaf node and return
                next_state, player = self.model.get_next_state(
                    current_node.state, current_node.player, action_edge[0])
                leaf_node = Node(next_state, player)
                action_edge[1].next_node = leaf_node

                return leaf_node, path

            current_node = next_node

    def evaluate(self, leaf_node):
        """Expand and evaluate leaf node.

        Args:
            leaf_node (object): Leaf node to expand and evaluate.

        Returns:
            (float): Node (state) value.
        """

        # Get valid actions
        valid_moves = self.model.get_valid_moves(leaf_node.state, leaf_node.player)

        edges = {}
        for m in valid_moves:
            edges[m] = Edge()

        leaf_node.expand(edges)

        # Get first elem in batch and value from result array
        return self.nn.predict(np.expand_dims(leaf_node.state, axis=0))[0][0]

    def backup(self, path, value):
        """Backup value to ancestry nodes.

        Args:
            path (list): List of edges that make path from start node to leaf node.
            value (float): Value to backup to all the edges on path.
        """

        return_t = value
        for edge in reversed(path):
            edge.update(return_t)
            return_t *= self.gamma
