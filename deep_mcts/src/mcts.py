import logging as log
import numpy as np

from abc import ABCMeta, abstractmethod
from humblerl import Mind
from tree.basic import Node


class MCTS(Mind, metaclass=ABCMeta):
    """MCTS search operations and planning logic."""

    def __init__(self, model, n_simulations=25):
        """Initialize MCTS object

        Args:
            model (Game): Perfect information dynamics/game.
            n_simulations (int): Number of simulations to perform before choosing action. (Default: 25)
        """

        self.model = model
        self.n_simulations = n_simulations

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

    def plan(self, state):
        """Conduct planning on state.

        Args:
            state (numpy.Array): State of game to plan on.

        Returns:
            numpy.Array: Planning result, normalized action probabilities.
            numpy.Array: Planning result, normalized action probabilities.
        """

        # Create root node if needed
        if self._root is None or self._root.state != state:
            log.info("Starting planning with NEW root node.")
            self._root = Node(state)
            _ = evaluate(self._root)
        else:
            log.info("Starting planning with OLD root node.")

        # Perform simulations
        for idx in range(self.n_simulations):
            log.info("Performing simulation number {}".format(idx + 1))

            # Simulate
            leaf, path = simulate(self._root)

            # Expand and evaluate
            value = evaluate(leaf)

            # Backup value
            backup(path, value)

        # Get actions' visit counts
        actions = np.zeros(self.model.get_action_size())
        for action, edge in self._root.edges.items():
            actions[action] = edge.num_visits

        # Calculate actions probabilities and return
        probs = actions / np.sum(actions)

        return probs, probs
