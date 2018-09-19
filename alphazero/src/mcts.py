import logging as log
import numpy as np

from abc import ABCMeta, abstractmethod
from third_party.humblerl import Callback, Mind
from tree.basic import Node

import time


class MCTS(Callback, Mind, metaclass=ABCMeta):
    """MCTS search operations and planning logic."""

    def __init__(self, model, nn, params={}):
        """Initialize MCTS object

        Args:
            model (Game): Perfect information dynamics/game.
            nn (NeuralNet): Artificial neural mind used to evaluate leaf states.
            params (dict): MCTS search hyper-parameters. Available:
              * 'n_simulations' (int) : Number of simulations to perform before choosing action.
                                        (Default: 25)

        Note:
            Add it as callback to humblerl loop to clear tree between episodes in train mode.
        """

        self.model = model
        self.nn = nn
        self.n_simulations = float(params.get('n_simulations', 25))
        self.timeout = float(params.get('timeout', "inf"))

        if self.n_simulations == self.timeout == float("inf"):
            raise Exception(
                "n_simulations and timeout cannot be set to inf simultaneously")

        self._tree = {}

    def _debug_log(self, root, player, metrics):
        # Evaluate root state
        pi, value = self.nn.predict(np.expand_dims(root.state, axis=0))

        # Log root state
        log.debug("Current player: %d", player)
        log.debug("Max search depth: %d", metrics['max_depth'])
        log.debug("Performed simulations: %d", metrics['simulations'])
        log.debug("Time: %d", metrics['simulation_time'])

        # Log MCTS root value and NN predicted value
        state_visits = 0
        state_value = 0
        for action, edge in root.edges.items():
            state_visits += edge.num_visits
            state_value += edge.qvalue * edge.num_visits

        log.debug("MCTS root value: %.5f", state_value / state_visits)
        log.debug("NN root value: %.5f\n", value[0])

        # Action size must be multiplication of board width
        BOARD_WIDTH = root.state.shape[1]
        action_size = self.model.get_action_size()
        if action_size % BOARD_WIDTH == 1:
            # There is extra 'null' action, ignore it
            # NOTE: For this WA to work 'null' action has to have last idx in the environment!
            action_size -= 1

        # Log MCTS actions scores and qvalues and NN prior probs
        visits = np.zeros(action_size)
        qvalues = np.zeros_like(visits)
        scores = np.zeros_like(visits)
        for action, edge in root.edges.items():
            visits[action] = edge.num_visits
            qvalues[action] = edge.qvalue
            scores[action] = edge.qvalue

        ucts = np.zeros_like(visits)
        for action, edge in root.edges.items():
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

    def expand(self, state, player):
        """Add new node to search tree.

        Args:
            state (np.array): Game state representation.
            player (int): Current (in passed state) player id.

        Return:
            Node: Node in search tree representing given state.

        Note:
            For now just store mapping in 'tree' dict from state.tostring() to Node. arrays are so
            small, that it'll have good performance:
            https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array

            If you'll deal with bigger arrays in the future e.g. from Atari games, consider wrapping
            it with own class with __hash__ and __eq__ implementation and in __hash_ convert
            to string only smaller part of original array. Allow python to deal with collisions.
        """

        # NOTE: In whole tree we store and work on only canonical board representations.
        #       Canonical means, that it's from perspective of CURRENT IN NODE player.
        #       From perspective of some player means that he is 1 on the board.
        canonical_state = self.model.get_canonical_form(state, player)

        node = Node(canonical_state)
        self._tree[canonical_state.tostring()] = node

        return node

    def clear_tree(self):
        """Empty search tree."""
        self._tree.clear()

    def query_tree(self, state, player):
        """Get node of given state from search tree.

        Args:
            state (np.array): Game state representation.
            player (int): Current (in passed state) player id.
        """

        # NOTE: In whole tree we store and work on only canonical board representations.
        #       Canonical means, that it's from perspective of CURRENT IN NODE player.
        #       From perspective of some player means that he is 1 on the board.
        canonical_state = self.model.get_canonical_form(state, player)

        return self._tree.get(canonical_state.tostring(), None)

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

        # Get/create root node
        root = self.query_tree(state, player)
        if root == None:
            root = self.expand(state, player)
            _ = self.evaluate(root, train_mode, is_root=True)

        # Perform simulations
        max_depth = 0
        max_simulations = self.n_simulations
        simulations = 0
        start_time = time.time()
        while time.time() < start_time + self.timeout and simulations < max_simulations:
            # Simulate
            simulations += 1
            leaf, path = self.simulate(root)

            # Keep max search depth
            max_depth = max(len(path), max_depth)

            # Expand and evaluate
            value = self.evaluate(leaf, train_mode)

            # Backup value
            # NOTE: Node higher in the tree is opponent node, invert value
            self.backup(path, -value)

        metrics = {"max_depth": max_depth, "simulations": simulations,
                   "simulation_time": time.time() - start_time}
        if debug_mode:
            self._debug_log(root, player, metrics)

        # Get actions' visit counts
        actions = np.zeros(self.model.get_action_size())
        for action, edge in root.edges.items():
            actions[action] = edge.num_visits

        return actions, metrics

    def on_episode_start(self, episode, train_mode):
        """Empty search tree between episodes if in train mode."""
        if train_mode:
            self.clear_tree()
        return {}
