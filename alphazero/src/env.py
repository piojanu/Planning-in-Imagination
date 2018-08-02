import numpy as np

from third_party.humblerl import Environment
from games import *  # This allows to create every game from board_games


class GameModel(object):
    """Wraps Game interface so you can use 0 for player one and 1 for player two.

    TODO (pj): When you will unify it to work with Atari games, move it to HumbleRL as an interface
               and this should be implementation of that interface.
    """

    def __init__(self, game):
        self.game = game

    def get_init_board(self):
        return self.game.getInitBoard()

    def get_board_size(self):
        board_shape = self.game.getBoardSize()
        if not isinstance(board_shape, (tuple, list)):
            board_shape = (board_shape,)
        return board_shape

    def get_action_size(self):
        return self.game.getActionSize()

    def get_next_state(self, board, player, action):
        gplayer = 1 if player == 0 else -1
        nboard, nplayer = self.game.getNextState(board, gplayer, action)

        return nboard, 0 if nplayer == 1 else 1

    def get_valid_moves(self, board, player):
        gplayer = 1 if player == 0 else -1
        valid_moves_map = self.game.getValidMoves(board, gplayer).astype(bool)
        valid_moves = np.arange(valid_moves_map.shape[0])[valid_moves_map]
        return valid_moves

    def get_game_ended(self, board, player):
        gplayer = 1 if player == 0 else -1
        return self.game.getGameEnded(board, gplayer)

    def string_representation(self, board):
        return self.game.stringRepresentation(board)

    def get_canonical_form(self, board, player):
        gplayer = 1 if player == 0 else -1
        return self.game.getCanonicalForm(board, gplayer)


class GameEnv(Environment):
    """Environment for board games from https://github.com/suragnair/alpha-zero-general

    Note:
        step(...) returns reward from perspective of player one!
    """

    def __init__(self, name):
        """Initialize Environment.

        Args:
            name (string): The name of game, in lower case. E.g. "connect4".
        """

        self.game = GameModel(eval(name)())
        self._last_action = -1
        self._last_player = -1

    @property
    def valid_actions(self):
        return self.game.get_valid_moves(self.current_state, self.player)

    def step(self, action):
        next_state, next_player = self.game.get_next_state(
            action=action, board=self.current_state, player=self.player)

        end = self.game.get_game_ended(next_state, self.player)
        # Current player took action, get reward from perspective of player one
        cannonical_reward = end * (1 if self.player == 0 else -1)
        # Draw has some small value, truncate it and leave only:
        # -1 (lose), 0 (draw/not finished yet), 1 (win)
        reward = float(int(cannonical_reward))

        self._last_action = action
        self._last_player = self.player

        self._curr_state = next_state
        self.player = next_player
        return next_state, next_player, reward, end != 0

    def reset(self, train_mode=True, first_player=0):
        self.train_mode = train_mode
        self.player = first_player
        # We need to represent init state from perspective of starting player.
        # Otherwise different first players could have different starting conditions e.g in Othello.
        self._curr_state = self.game.get_canonical_form(self.game.get_init_board(), self.player)
        return self._curr_state, self.player

    def render(self, fancy=False):
        """Display board when environment is in test mode.

        Args:
            fancy (bool): Display a fancy 2D board.
        """

        print("Player {}, Action {}".format(self._last_player, self._last_action))
        if fancy and self.current_state.ndim == 2:
            self.render_fancy_board()
        else:
            print(self.current_state)

    def render_fancy_board(self):
        def line_sep(length):
            print(" ", end="")
            for _ in range(length):
                print("=", end="")
            print("")

        state = self.current_state.astype(int)
        m, n = state.shape
        line_sep(3 * n + 1)
        legend = {1: "X", -1: "O"}
        for i in range(m):
            print("|", end=" ")
            for j in range(n):
                s = legend.get(state[i][j], "-")
                if (i * m + j) == self._last_action:
                    print("\033[1m{:2}\033[0m".format(s), end=" ")
                else:
                    print("{:2}".format(s), end=" ")
            print("|")
        line_sep(3 * n + 1)
