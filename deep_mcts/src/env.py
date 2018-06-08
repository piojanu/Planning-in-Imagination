from humblerl import Environment
from src.games import *  # This allows to create every game from board_games


class GameEnv(Environment):
    def __init__(self, game_name):
        self.game = eval(game_name)()
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
        end = self.game.getGameEnded(self.current_state, self.player)
        self.player = next_player
        return next_state, end, end != 0

    def reset(self, train_mode):
        self.player = 0
        return self.game.getInitBoard()
