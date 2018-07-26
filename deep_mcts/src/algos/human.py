import numpy as np

import humblerl as hrl


class HumanPlayer(hrl.Mind):
    """Mind representing an agent controlled by a human player."""

    def __init__(self, game):
        """
        Args:
            game (Game): Game model needed to check available actions at given state.
        """
        self.game = game

    def plan(self, state, player, train_mode=True, debug_mode=False):
        # Get valid actions
        valid_actions = self.game.get_valid_moves(state, player)

        if valid_actions.size == 1 and state.size in valid_actions:
            # You have no other available actions then skip move
            logits = np.zeros(self.game.get_action_size(), dtype=np.float32)
            logits[-1] = 1.0
        else:
            # Create a board with available actions represented as int values from 1 to n_actions.
            # Invalid actions are representede with NaN, which is rendered as "-".
            available_actions_board = np.full(state.shape, np.nan)
            np.set_printoptions(nanstr="-")
            valid_actions_board = np.add(valid_actions, 1)
            np.put(available_actions_board, valid_actions, valid_actions_board)

            print("\nAvailable actions:")
            print("------------------")
            print(available_actions_board)

            action = -1
            while action not in valid_actions_board:
                action_str = input("Action: ")
                try:
                    action = int(action_str)
                    if action not in valid_actions_board:
                        print("Given action is not available! Please input a valid action! Available: {}"
                              .format(valid_actions_board))
                except ValueError:
                    print("Please input a valid integer!")

            # Create "logits" with all elements equal to 0, and taken action equal to 1.
            logits = np.zeros(self.game.get_action_size(), dtype=np.float32)
            logits[np.where(available_actions_board.flatten() == action)] = 1.0

        return logits, {}
