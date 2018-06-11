import numpy as np

from humblerl import Environment
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


class GameEnv(Environment):
    """Environment for board games from https://github.com/suragnair/alpha-zero-general

    Note:
        Available games:
          * connect4
    """

    def __init__(self, name):
        """Initialize Environment.

        Args:
            name (string): The name of game, in lower case. E.g. "connect4".
        """

        self.game = GameModel(eval(name)())

    def step(self, action):
        next_state, next_player = self.game.get_next_state(
            action=action, board=self.current_state, player=self.player)
        end = self.game.get_game_ended(next_state, self.player)
        reward = float(int(end * (-1) ** self.player))
        self.player = next_player
        self._curr_state = next_state
        self._display()
        return next_state, next_player, reward, end != 0

    def reset(self, train_mode):
        self.train_mode = train_mode
        self.player = 0
        self._curr_state = self.game.get_init_board()
        self._display()
        return self._curr_state, self.player

    @property
    def valid_actions(self):
        return self.game.get_valid_moves(self.current_state, self.player)

    def _display(self):
        """Display board when environment is in test mode.
        """
        if not self.train_mode:
            print("<---")
            print(' '.join(map(str, range(len(self.current_state[0])))))
            print(self.current_state)
            print("<---")
