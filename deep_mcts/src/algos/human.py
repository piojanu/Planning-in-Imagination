import numpy as np

import humblerl as hrl


class HumanPlayer(hrl.Mind):
    """Mind representing an agent controlled by a human player."""

    def __init__(self, env):
        """
        Args:
            env (Environment): Environment to take actions in. Needed to check
                               available actions at given state.
        """
        self.env = env

    def plan(self, state, player, train_mode=True, debug_mode=False):
        valid_actions = self.env.valid_actions
        action_range = range(1, len(valid_actions) + 1)

        # Create a board with available actions represented as int values from 1 to n_actions.
        # Invalid actions are representede with NaN, which is rendered as "-".
        available_actions_board = np.full(state.shape, np.nan)
        np.set_printoptions(nanstr="-")
        np.put(available_actions_board, valid_actions, action_range)

        print("\nAvailable actions:")
        print("------------------")
        print(available_actions_board)
        print("\nPlease choose one of available actions: {}".format(list(action_range)))

        action = -1
        while action not in action_range:
            action_str = input("Action: ")
            try:
                action = int(action_str)
                if action not in action_range:
                    print("Given action is not available! Please input a valid action!"
                          .format(action_range))
            except ValueError:
                print("Please input a valid integer!".format(action_range))

        # Create "logits" with all elements equal to 0, and taken action equal to 1.
        logits = np.zeros_like(state, dtype=np.float32)
        logits[np.where(available_actions_board == action)] = 1.0
        return logits, {}
