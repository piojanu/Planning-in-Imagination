import logging as log

from abc import ABCMeta, abstractmethod
from humblerl import Policy
from tree.basic import Node


class MCTS(metaclass=ABCMeta):
    """MCTS search operations."""

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


class Plan(Policy):
    """It conduct planning using MCTS operations."""

    def __init__(self, mcts, n_simulations=25):
        """Initialize Plan object.

        Args:
            mcts (MCTS): Monte-Carlo Tree Search algorithm operations.
            simulations (int): Number of simulations to perform before choosing action (Default: 25)
        """

        self.mcts = mcts
        self.n_simulations = n_simulations

        self._root = None

    def __call__(self, state):
        """Conduct planning and choose action.

        Args:
            state (object): Current world state to start from.

        Returns:
            int: action to take in the environment.
        """

        # Create root node if needed
        if self._root is None or self._root.state != state:
            log.info("Starting planning with NEW root node.")
            self._root = Node(state)
            _ = self.mcts.evaluate(self._root)
        else:
            log.info("Starting planning with OLD root node.")

        # Perform simulations
        for idx in range(self.n_simulations):
            log.info("Performing simulation number {}".format(idx + 1))

            # Simulate
            leaf, path = self.mcts.simulate(self._root)

            # Expand and evaluate
            value = self.mcts.evaluate(leaf)

            # Backup value
            self.mcts.backup(path, value)

        # Get actions' visit counts
        actions_visits = {}
        for action, edge in self._root.edges.items():
            actions_visits[action] = edge.num_visits

        # TODO (by piojanu): You should implement distribution with temperature in here
        # for exploratory play.

        # Choose action deterministically
        action = max(actions_visits, key=actions_visits.get)
        self._root = self._root.edges[action].next_node

        return action
