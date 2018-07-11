import logging as log
import numpy as np

from mcts import MCTS
from tree.basic import Edge, Node


class Planner(MCTS):
    """AlphaZero search operations and planning logic."""

    def __init__(self, model, nn, params={}):
        """Initialize MCTS object

        Args:
            model (Game): Perfect information dynamics/game.
            nn (NeuralNet): Artificial neural mind used to evaluate leaf states.
            params (dict): MCTS search hyper-parameters. Available:
              * 'c' (float)           : UCT exploration-exploitation trade-off param (Default: 1.)
              * 'dirichlet_noise'     : Dirichlet noise added to root prior (Default: 0.03)
              * 'noise_ratio'          : Noise contribution to prior probabilities (Default: 0.25)
              * 'gamma' (float)       : Discounting factor for value. (Default: 1./no discounting)
              * 'n_simulations' (int) : Number of simulations to perform before choosing action.
                                        (Default: 25)
        """

        MCTS.__init__(self, model, nn, params)

        self.c = params.get('c', 1.)
        self.gamma = params.get('gamma', 1.)
        self.dirichlet_noise = params.get('dirichlet_noise', 0.03)
        self.noise_ratio = params.get('noise_ratio', 0.25)

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
            action, edge = action_edge

            path.append(edge)
            next_node = edge.next_node

            if next_node is None:
                # This edge wasn't explored yet, create leaf node and return
                next_state, player = self.model.get_next_state(
                    current_node.state, current_node.player, action)
                leaf_node = Node(next_state, player)
                edge.next_node = leaf_node

                return leaf_node, path

            current_node = next_node

    def evaluate(self, leaf_node, train_mode, is_root=False):
        """Expand and evaluate leaf node.

        Args:
            leaf_node (object): Leaf node to expand and evaluate.
            train_mode (bool): Informs whether add additional Dirichlet noise for exploration.
            is_root (bool): Whether this is tree root. (Default: False)

        Returns:
            (float): Node (state) value.
        """

        # Get relative state
        relative_state = self.model.get_canonical_form(leaf_node.state, leaf_node.player)
        # Evaluate state
        pi, value = self.nn.predict(np.expand_dims(relative_state, axis=0))

        # Take first element in batch
        pi = pi[0]
        value = value[0][0]

        # Change value to game result if this is terminal state
        end_result = self.model.get_game_ended(leaf_node.state, leaf_node.player)
        if end_result != 0:
            value = end_result

        # Add Dirichlet noise to root node prior
        if is_root and train_mode:
            pi = (1 - self.noise_ratio) * pi + self.noise_ratio * \
                np.random.dirichlet([self.dirichlet_noise, ] * len(pi))

        # Get valid actions probabilities
        valid_moves = self.model.get_valid_moves(leaf_node.state, leaf_node.player)

        # Create edges of this node
        edges = {}

        # Renormalize valid actions probabilities, but only if there are any valid actions
        if len(valid_moves) != 0:
            probs = np.zeros_like(pi)
            probs[valid_moves] = pi[valid_moves]
            sum_probs = np.sum(probs)
            if sum_probs <= 0:
                # If all valid moves were masked make all valid moves equally probable
                log.warn("All valid moves were masked, do workaround!")
                probs[valid_moves] = 1
            probs = probs / sum_probs

            # Fill this node edges with priors
            for m in valid_moves:
                edges[m] = Edge(prior=probs[m])

        # Expand node with edges
        leaf_node.expand(edges)

        return value

    def backup(self, path, value):
        """Backup value to ancestry nodes.

        Args:
            path (list): List of edges that make path from start node to leaf node.
            value (float): Value to backup to all the edges on path.
        """

        return_t = value
        for edge in reversed(path):
            edge.update(return_t)
            # NOTE: Node higher in tree is opponent node, negate value
            return_t *= -self.gamma
