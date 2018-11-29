from common_utils import Storage, get_configs


class Config(object):
    """Loads custom configuration, unspecified parameters are taken from default configuration.

    Args:
        config_path (str): Path to .json file with custom configuration
        is_debug (bool): Specify to enable debugging features
        allow_render (bool): Specify to enable render/plot features
    """

    def __init__(self, config_path, is_debug, allow_render):
        default_config, custom_config = get_configs(config_path)

        # Merging default and custom configs, for repeating keys second dict overwrites values
        self.general = {**default_config["general"], **custom_config.get("general", {})}
        self.vae = {**default_config["vae_training"], **custom_config.get("vae_training", {})}
        self.rnn = {**default_config["rnn_training"], **custom_config.get("rnn_training", {})}
        self.ctrl = {**default_config["ctrl_play"], **custom_config.get("ctrl_play", {})}
        self.is_debug = is_debug
        self.allow_render = allow_render


class ExperienceStorage(Storage):
    """Record whole trajectories and allow for save to/load from Pickle file.

    Each trajectory (list) keeps transitions (tuples) and such consists of in order:
    * State,
    * Action,
    * Reward,
    * If this is terminal transition flag,
    * Action probabilities estimated by MCTS,
    * State value.

    Note:
        Experience replay capacity is given in number of games.
    """

    def on_step_taken(self, step, transition, info):
        self.small_bag.append((
            transition.state[0],  # Take latent state, discard hidden state
            transition.action,    # Action is needed as RNN input
            transition.reward,
            transition.is_terminal,
            self._recent_action_probs
        ))

        if transition.is_terminal:
            # NOTE: We just need next state to train RNN predict it, any other value doesn't matter.
            tau = [(transition.next_state[0], None, None, None, None, None)]

            return_t = 0
            for state, action, reward, is_terminal, mcts_pi in reversed(self.small_bag):
                return_t = reward + self.gamma * return_t
                tau.insert(0, (state, action, reward, is_terminal, mcts_pi, return_t))

            self.big_bag.append(tau)
            if len(self.big_bag) > self.exp_replay_size:
                self.big_bag.popleft()

            self.small_bag.clear()
