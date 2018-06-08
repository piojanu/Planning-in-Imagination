import numpy as np

from mcts import MCTS
from tree.basic import Node, Edge


class Planner(MCTS):
    """MCTS search operations and planning logic."""

    def __init__(self, model, params={}):
        """Initialize MCTS object

        Args:
            model (Game): Perfect information dynamics/game.
            params (dict): MCTS search hyper-parameters. Available:
              * 'n_simulations' (int): Number of simulations to perform before choosing action.
                                       (Default: 25)
        """

        MCTS.__init__(self, model, params)
        self.n_simulations = 1

    def simulate(self, start_node):
        """Search through tree from start node to leaf.

        Args:
            start_node (Node): Where to start the search.

        Returns:
            (Node): Leaf node.
            (list): List of edges that make path from start node to leaf node.
        """

        return self._root, None

    def evaluate(self, leaf_node):
        """Expand and evaluate leaf node.

        Args:
            leaf_node (object): Leaf node to expand and evaluate.

        Returns:
            (float): Node (state) value.
        """

        state = leaf_node.state
        player = leaf_node.player

        valid_moves_map = self.model.get_valid_moves(state, player)
        valid_moves = np.arange(valid_moves_map.shape[0])[valid_moves_map]

        edges = {}
        for i, m in enumerate(valid_moves):
            edges[m] = Edge(i)
            edges[m]._num_visits = i

        leaf_node.expand(edges)
        return None

    def backup(self, path, value):
        """Backup value to ancestry nodes.

        Args:
            path (list): List of edges that make path from start node to leaf node.
            value (float): Value to backup to all the edges on path.
        """

        return None, None
