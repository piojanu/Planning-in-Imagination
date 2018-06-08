from humblerl import Environment
from src.games import *  # This allows to create every game from board_games


class GameEnv(Environment):
    def __init__(self, name):
        self.game = eval(name)()
        self.reset()
        self._action_space_info = Environment.ActionSpaceInfo(
            size=self.game.getActionSize(),
            type=Environment.DISCRETE_SPACE,
            descriptions=None
        )
        self._state_space_info = Environment.StateSpaceInfo(
            size=self.game.getBoardSize(),
            type=Environment.DISCRETE_SPACE
        )

    def step(self, action):
        next_state, next_player = self.game.getNextState(
            action=action, board=self.current_state, player=self.player)
        end = self.game.getGameEnded(next_state, self.player)
        self.player = next_player
        self._curr_state = next_state
        decoded_player = self._decode_player(self.player)
        return next_state, decoded_player, end, end != 0

    def reset(self, train_mode):
        self.player = 1
        self._curr_state = self.game.getInitBoard()
        return self._curr_state, self._decode_player(self.player)

    def _decode_player(self, player):
        return 0 if player == 1 else -1
