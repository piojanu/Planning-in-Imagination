import datetime as dt
import glob
import os.path

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
        self.planner = {**default_config["planner"], **custom_config.get("planner", {})}
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
            transition.state.latent,  # Take latent state, discard hidden state
            transition.action,    # Action is needed as RNN input
            transition.reward,
            transition.is_terminal,
            self._recent_action_probs
        ))

        if transition.is_terminal:
            # NOTE: We just need next state to train RNN predict it, any other value doesn't matter.
            tau = [(transition.next_state.latent, None, None, None, None, None)]

            return_t = 0
            for state, action, reward, is_terminal, mcts_pi in reversed(self.small_bag):
                return_t = reward + self.gamma * return_t
                tau.insert(0, (state, action, reward, is_terminal, mcts_pi, return_t))

            self.big_bag.append(tau)
            if len(self.big_bag) > self.exp_replay_size:
                self.big_bag.popleft()

            self.small_bag.clear()


def create_checkpoint_path(ckpt_dir, iteration, epoch, score):
    """Encode iter. num., epoch num., score and date and time into path."""
    filename = "_".join(['memory',
                         '{0:03d}'.format(iteration),
                         '{0:04d}'.format(epoch),
                         '{0:09.5f}'.format(score),
                         dt.datetime.now().strftime("%d-%mT%H:%M")]) + ".ckpt"
    return os.path.join(ckpt_dir, filename)


def read_checkpoint_metadata(path):
    """Read iter. num., epoch num. and score from filename."""
    filename = os.path.basename(path)
    iteration, epoch, score = filename.split('_')[1:-1]
    return int(iteration), int(epoch), float(score)


def get_last_checkpoint_path(ckpt_dir):
    """Looks for files with `prefix` in filename and '.ckpt' extension."""
    files = list(filter(os.path.isfile, glob.glob(os.path.join(ckpt_dir, '*.ckpt'))))
    files.sort(key=lambda x: read_checkpoint_metadata(x)[0])

    return files[-1] if len(files) > 0 else None
