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

    def _debug_log(self, max_depth):
        # Evaluate root state
        relative_state = self.model.get_canonical_form(self._root.state, self._root.player)
        pi, value = self.nn.predict(np.expand_dims(relative_state, axis=0))

        # Log root state
        log.debug("Current player: %d", self._root.player)
        log.debug("Max search depth: %d", max_depth)

        # Log MCTS root value and NN predicted value
        state_visits = 0
        state_value = 0
        for action, edge in self._root.edges.items():
            state_visits += edge.num_visits
            state_value += edge.qvalue * edge.num_visits

        log.debug("MCTS root value: %.5f", state_value / state_visits)
        log.debug("NN root value: %.5f\n", value[0])

        # Action size must be multiplication of board width
        BOARD_WIDTH = self._root.state.shape[1]
        action_size = self.model.get_action_size()
        if action_size % BOARD_WIDTH == 1:
            # There is extra 'null' action, ignore it
            # NOTE: For this WA to work 'null' action has to have last idx in the environment!
            action_size -= 1

        # Log MCTS actions scores and qvalues and NN prior probs
        visits = np.zeros(action_size)
        qvalues = np.zeros_like(visits)
        scores = np.zeros_like(visits)
        for action, edge in self._root.edges.items():
            visits[action] = edge.num_visits
            qvalues[action] = edge.qvalue
            scores[action] = edge.qvalue

        ucts = np.zeros_like(visits)
        for action, edge in self._root.edges.items():
            ucts[action] = self.c * edge.prior * \
                np.sqrt(1 + np.sum(visits)) / (1 + edge.num_visits)
            scores[action] += ucts[action]

        log.debug("Prior probabilities:\n%s\n", np.array2string(
            pi[0][:action_size].reshape([-1, BOARD_WIDTH]), formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Exploration bonuses:\n%s\n", np.array2string(
            ucts.reshape([-1, BOARD_WIDTH]), formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Actions qvalues:\n%s\n", np.array2string(
            qvalues.reshape([-1, BOARD_WIDTH]), formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Actions scores:\n%s\n", np.array2string(
            scores.reshape([-1, BOARD_WIDTH]), formatter={'float_kind': lambda x: "%.5f" % x}))
        log.debug("Actions visits:\n%s\n", visits.reshape([-1, BOARD_WIDTH]))

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
    def evaluate(self, leaf_node, train_mode, is_root=False):
        """Expand and evaluate leaf node.

        Args:
            leaf_node (object): Leaf node to expand and evaluate.
            train_mode (bool): Informs Neural Net whether it's in training or evaluation mode.
            is_root (bool): Whether this is tree root. (Default: False)

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

    def plan(self, state, player, train_mode, debug_mode):
        """Conduct planning on state.

        Args:
            state (numpy.Array): State of game to plan on.
            player (int): Current player index.
            train_mode (bool): Informs planner whether it's in training mode and should enable
        additional exploration.
            debug_mode (bool): Informs planner whether it's in debug mode or not.

        Returns:
            numpy.Array: Planning result, unnormalized action probabilities.
            dict: Planning metrics.
        """

        # Create root node
        # TODO (pj): Implement reusing subtrees in next moves.
        self._root = Node(state, player)
        _ = self.evaluate(self._root, train_mode, is_root=True)

        # Perform simulations
        max_depth = 0
        for idx in range(self.n_simulations):
            # Simulate
            leaf, path = self.simulate(self._root)

            # Keep max search depth
            max_depth = max(len(path), max_depth)

            # Expand and evaluate
            value = self.evaluate(leaf, train_mode)

            # Backup value
            # NOTE: Node higher in the tree is opponent node, invert value
            self.backup(path, -value)

        if debug_mode:
            self._debug_log(max_depth)

        # Get actions' visit counts
        actions = np.zeros(self.model.get_action_size())
        for action, edge in self._root.edges.items():
            actions[action] = edge.num_visits

        return actions, {"max_depth", max_depth}
