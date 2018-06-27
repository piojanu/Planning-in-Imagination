import logging as log
import numpy as np

from .value_function import Planner as VFPlanner
from tree.basic import Edge


class Planner(VFPlanner):
    """AlphaZero search operations and planning logic."""

    def __init__(self, model, nn, params={}):
        """Initialize MCTS object

        Args:
            model (Game): Perfect information dynamics/game.
            nn (NeuralNet): Artificial neural mind used to evaluate leaf states.
            params (dict): MCTS search hyper-parameters. Available:
              * 'c' (float)           : UCT exploration-exploitation trade-off param (Default: 1.)
              * 'dirichlet_noise'     : Dirichlet noise added to root prior (Default: 0.03)
              * 'noise_rate'          : Noise contribution to prior probabilities (Default: 0.25)
              * 'gamma' (float)       : Discounting factor for value. (Default: 1./no discounting)
              * 'n_simulations' (int) : Number of simulations to perform before choosing action.
                                        (Default: 25)
        """

        VFPlanner.__init__(self, model, nn, params)

        self.dirichlet_noise = params.get('dirichlet_noise', 0.03)
        self.noise_rate = params.get('noise_rate', 0.25)

    def _log_debug(self):
        # Evaluate root state
        relative_state = self.model.get_canonical_form(self._root.state, self._root.player)
        pi, value = self.nn.predict(np.expand_dims(relative_state, axis=0))

        # Log MCTS actions scores and qvalues and NN prior probs
        scores = np.zeros(self.model.get_action_size())
        qvalues = np.zeros(self.model.get_action_size())
        for action, edge in self._root.edges.items():
            scores[action] = edge.num_visits
            qvalues[action] = edge.qvalue

        log.debug("Actions scores:\n%s\n", scores[:-1].reshape([-1, self._root.state.shape[1]]))
        log.debug("Actions qvalues:\n%s\n", np.array2string(
            qvalues[:-1].reshape([-1, self._root.state.shape[1]]),
            formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Prior probabilities:\n%s\n", np.array2string(
            pi[0, :-1].reshape([-1, self._root.state.shape[1]]),
            formatter={'float_kind': lambda x: "%.5f" % x}))

        # Log MCTS root value and NN predicted value
        state_visits = 0
        state_value = 0
        for action, edge in self._root.edges.items():
            state_visits += edge.num_visits
            state_value += edge.qvalue * edge.num_visits

        log.debug("MCTS root value: %.5f", state_value / state_visits)
        log.debug("NN root value: %.5f\n", value[0])

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
        value = value[0]

        # Add Dirichlet noise to root node prior
        if is_root and train_mode:
            pi = (1 - self.noise_rate) * pi + self.noise_rate * \
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

        # Get value from result array
        return value[0]
