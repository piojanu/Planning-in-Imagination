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

        self._debug_last_cmd = "s"
        self._debug_run = False
        self._debug_continue = False

    @abstractmethod
    def _log_debug(self):
        pass

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

    def plan(self, state, player, train_mode):
        """Conduct planning on state.

        Args:
            state (numpy.Array): State of game to plan on.
            player (int): Current player index.
            train_mode (bool): Informs planner whether it's in training mode and should enable
        additional exploration. (Default: True)

        Returns:
            numpy.Array: Planning result, unnormalized action probabilities.
        """

        # Create root node
        # TODO (pj): Implement reusing subtrees in next moves.
        self._root = Node(state, player)
        _ = self.evaluate(self._root, train_mode, is_root=True)

        # Perform simulations
        for idx in range(self.n_simulations):
            # Simulate
            leaf, path = self.simulate(self._root)

            # Expand and evaluate
            value = self.evaluate(leaf, train_mode)

            # Backup value
            # NOTE: Node higher in the tree is opponent node, invert value
            self.backup(path, -value)

            if log.getLogger(__name__).isEnabledFor(log.DEBUG):
                # Debug menu:
                if not self._debug_continue and not self._debug_run:
                    cmd = input(
                        "\nPlayer {} | Run this player (r), Continue to next planning (c), Step simulation (s): ".format(
                            player))
                    if cmd == "":
                        cmd = self._debug_last_cmd

                    if cmd == "r":
                        self._debug_run = True
                        continue
                    elif cmd == "c":
                        self._debug_continue = True
                        continue

                    self._debug_last_cmd = cmd

                    log.debug("##### Simulation num.: %d #####\n", idx + 1)
                    self._log_debug()

        # Finished simulation, reset debug continue flag and print final log
        if log.getLogger(__name__).isEnabledFor(log.DEBUG):
            # If simulation skipped log final debug log
            if self._debug_continue or self._debug_run:
                self._log_debug()

            self._debug_continue = False

        # Get actions' visit counts
        actions = np.zeros(self.model.get_action_size())
        for action, edge in self._root.edges.items():
            actions[action] = edge.num_visits

        return actions
