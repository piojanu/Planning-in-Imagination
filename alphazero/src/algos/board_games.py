from callbacks import Storage
from third_party.humblerl import Callback, Mind, Vision


class BoardGameMind(Mind):
    """Wraps two minds and dispatch work to appropriate one based on player id in state."""

    def __init__(self, one, two):
        """Initialize board game mind.

        Args:
            one (Mind): Mind which will plan for player "1".
            two (Mind): Mind which will plan for player "-1".
        """

        # Index '1' for player one, index '-1' for player two
        self.players = [None, one, two]

    def plan(self, state, train_mode, debug_mode):
        """Conduct planning on state.

        Args:
            state tuple(numpy.ndarray, int): State of game to plan on and current player id.
            train_mode (bool): Informs planner whether it's in training mode and should enable
                additional exploration.
            debug_mode (bool): Informs planner whether it's in debug mode or not.

        Returns:
            np.ndarray: Planning result, unnormalized action probabilities.
            dict: Planning metrics.
        """

        board, player = state
        return self.players[player].plan(board, train_mode, debug_mode)


class BoardGameVision(Vision):
    """Transforms board game state and reward to canonical one."""

    def __init__(self, game):
        """Initialize board game vision.

        Args:
            game (Game): Board game object.
        """

        self.game = game

    def __call__(self, state, reward=0.):
        """Transform board game state and reward:

        Args:
            state (tuple): Board and player packed in tuple.
            reward (float): Transition reward. (Default: 0.)

        Returns:
            state (np.ndarray): Canonical board game (from perspective of current player).
            reward (float): Canonical transition reward (from perspective of current player).
        """

        board, player = state
        cannonical_state = self.game.getCanonicalForm(board, player)
        # WARNING! SHIT CODE... Please refactor if you have better idea.
        # Reward from env is from player one perspective, so we multiply reward by player
        # id which is 1 for player one or -1 player two. We also multiply by -1 because this is
        # id of "next player", and we want to represent reward from perspective of current player.
        cannonical_reward = reward * player * -1

        return (cannonical_state, player), cannonical_reward


class BoardGameRender(Callback):
    def __init__(self, env, render, fancy=False):
        self.env = env
        self.render = render
        self.fancy = fancy

    def on_step_taken(self, step, transition, info):
        self.do_render()

    def do_render(self):
        if self.render:
            self.env.render(self.fancy)


class BoardGameStorage(Storage):
    """Wraps Storage callback to unpack state from board game."""

    def _create_small_package(self, transition):
        return (transition.state[0], transition.reward, self._recent_action_probs)


class BoardGameTournament(Callback):
    def __init__(self):
        self.reset()

    def on_loop_start(self):
        self.reset()

    def on_step_taken(self, step, transition, info):
        if transition.is_terminal:
            # NOTE: Because players have fixed player id, and reward is returned from perspective
            #       of current player, we transform it into perspective of player one and check
            #       who wins.
            reward = transition.state[1] * transition.reward
            if reward == 0:
                self.draws += 1
            elif reward > 0:
                self.wannabe += 1
            else:
                self.best += 1

    def reset(self):
        self.wannabe, self.best, self.draws = 0, 0, 0

    @property
    def metrics(self):
        return {"wannabe": self.wannabe, "best": self.best, "draws": self.draws}

    @property
    def results(self):
        return self.wannabe, self.best, self.draws
