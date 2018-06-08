import numpy as np

from abc import ABCMeta, abstractmethod


class Node(object):
    """Represents state in MCTS search tree.

    Note:
        Node object is immutable.
    """

    def __init__(self, state, player):
        """Initialize Node object with state.

        Args:
            state (object): The environment state corresponding to this node in the search tree.
            player (int): Current player index.

        Note:
            Node is left without exit edges (empty dict) when it's terminal.
        """

        self._state = state
        self._player = player
        self._edges = None

    @property
    def state(self):
        """object: The environment state corresponding to this node in the search tree."""
        return self._state

    @property
    def player(self):
        """int: This node player index."""
        return self._state

    @property
    def edges(self):
        """list of Edges: Mapping from this node's possible actions to corresponding edges."""
        return self._edges

    def expand(self, edges):
        """Initialize Node object with edges.

        Args:
            edges (dict of Edges): Mapping from this node's possible actions to corresponding edges.
        """

        self._edges = edges

    def select_edge(self, c=1.):
        """Choose next action (edge) according to UCB formula.

        Args:
            c (float): The parameter c >= 0 controls the trade-off between choosing lucrative nodes
        (low c) and exploring nodes with low visit counts (high c). (Default: 1)

        Returns:
            int: Action chosen with UCB formula.
            Edge: Edge which represents proper action chosen with UCB formula.

            or

            None: If it is terminal node and has no exit edges.
        """

        assert self.edges is not None, "This node hasn't been expanded yet!"

        if len(self.edges) == 0:
            return None

        state_visits = 0
        scores = {}

        # Initialize every edge's score to its Q-value and count current state visits
        for action, edge in self.edges.items():
            state_visits += edge.num_visits
            scores[(action, edge)] = edge.qvalue

        # Add exploration term to every edge's score
        for action, edge in self.edges.items():
            scores[(action, edge)] += c * np.sqrt(np.log(1 + state_visits) / (1 + edge.num_visits))

        # Choose next action and edge with highest score
        action_edge = max(scores, key=scores.get)
        return action_edge


class Edge(object):
    """Represents state-actions pair in MCTS search tree."""

    def __init__(self, qvalue=0., next_node=None):
        """Initialize Edge object.

        Args:
            qvalue (float): Initial Q-value of this state-action pair. (Default: 0.)
            next_node (Node): Next node this state-action leads to.
        If None, it means that this edge wasn't explored yet (Default: None).
        """

        self.next_node = next_node
        self._qvalue = qvalue
        self._num_visits = 0

    @property
    def qvalue(self):
        """float: Current Q-value estimate of this state-action pair."""
        return self._qvalue

    @property
    def num_visits(self):
        """int: Number of times this state-action pair was visited."""
        return self._num_visits

    def update(self, return_t):
        """Update edge with data from child.

        Args:
            return_t (float): Possibly discounted backup return to timestep 't' (this edge).
        """

        # This is formula for iteratively calculating average
        # NOTE: You can check that first arbitrary value will be forgotten after fist update
        self._qvalue += (return_t - self.qvalue) / (self.num_visits + 1)
        self._num_visits += 1
