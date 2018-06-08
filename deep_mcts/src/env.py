from humblerl import Environment
from games import *  # This allows to create every game from board_games


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

        self.game = eval(name)()

    def step(self, action):
        next_state, next_player = self.game.getNextState(
            action=action, board=self.current_state, player=self.player)
        end = self.game.getGameEnded(next_state, self.player)
        reward = end * self.player
        self.player = next_player
        self._curr_state = next_state
        decoded_player = self._decode_player(self.player)
        self._display()
        return next_state, decoded_player, reward, end != 0

    def reset(self, train_mode):
        self.train_mode = train_mode
        self.player = 1
        self._curr_state = self.game.getInitBoard()
        self._display()
        return self._curr_state, self._decode_player(self.player)

    def _decode_player(self, player):
        """Translate Game interface specific index to normalized value.
        The rules are as follows:
            1  -> 0
            -1 -> 1

        Args:
            player (int): Encoded player index.

        Returns:
            (int): Decoded player index.
        """
        return 0 if player == 1 else -1

    def _display(self):
        """Display board when environment is in test mode.
        """
        if not self.train_mode:
            print(" -----------------------")
            print(' '.join(map(str, range(len(self.current_state[0])))))
            print(self.current_state)
            print(" -----------------------")
