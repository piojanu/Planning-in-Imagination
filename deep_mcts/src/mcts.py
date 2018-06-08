import logging as log
import numpy as np

from abc import ABCMeta, abstractmethod
from humblerl import Mind
from tree.basic import Node


class MCTS(Mind, metaclass=ABCMeta):
    """MCTS search operations and planning logic."""

    def __init__(self, model, nn, params={}):
        """Initialize MCTS object

        Args:
            model (Game): Perfect information dynamics/game.
            nn (NeuralNet): Artificial neural mind used to evaluate leaf states.
            params (dict): MCTS search hyper-parameters. Available:
              * 'n_simulations' (int) : Number of simulations to perform before choosing action.
                                        (Default: 25)
        """

        self.model = model
        self.nn = nn
        self.n_simulations = params.get('n_simulations', 25)

        self._root = None

    @abstractmethod
    def simulate(self, start_node):
        """Search through tree from start node to leaf.

        Args:
            start_node (Node): Where to start the search.

        Returns:
            (Node): Leaf node.
            (list): List of edges that make path from start node to leaf node.
        """

        pass

    @abstractmethod
    def evaluate(self, leaf_node):
        """Expand and evaluate leaf node.

        Args:
            leaf_node (object): Leaf node to expand and evaluate.

        Returns:
            (float): Node (state) value.
        """

        pass

    @abstractmethod
    def backup(self, path, value):
        """Backup value to ancestry nodes.

        Args:
            path (list): List of edges that make path from start node to leaf node.
            value (float): Value to backup to all the edges on path.
        """

        pass

    def plan(self, state, player):
        """Conduct planning on state.

        Args:
            state (numpy.Array): State of game to plan on.
            player (int): Current player index.

        Returns:
            numpy.Array: Planning result, normalized action probabilities.
            numpy.Array: Planning result, normalized action probabilities.
        """

        # Create root node
        # TODO (pj): Implement reusing subtrees in next moves.
        self._root = Node(state, player)
        _ = self.evaluate(self._root)

        # Perform simulations
        for idx in range(self.n_simulations):
            log.info("Performing simulation number {}".format(idx + 1))

            # Simulate
            leaf, path = self.simulate(self._root)

            # Expand and evaluate
            value = self.evaluate(leaf)

            # Backup value
            self.backup(path, value)

        # Get actions' visit counts
        actions = np.zeros(self.model.get_action_size())
        for action, edge in self._root.edges.items():
            actions[action] = edge.num_visits

        # Calculate actions probabilities and return
        probs = actions / np.sum(actions)

        return probs, probs
