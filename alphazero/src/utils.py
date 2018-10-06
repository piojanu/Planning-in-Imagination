import datetime as dt
import glob
import os
import tensorflow as tf
import json
from env import GameEnv


class TensorBoardLogger(object):
    """Logging in TensorBoard without TensorFlow ops.

    https://gist.github.com/1f8dfb1b5c82627ae3efcfbbadb9f514.git
    Simple example on how to log scalars and images to tensorboard without tensor ops.

    License: Copyleft
    Author: Michael Gygli
    """

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Args:
            tag (basestring): Name of the scalar.
            value (number): Value to log.
            step (int): Training iteration.
        """

        summary = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()


class Config(object):
    def __init__(self, config, debug=False):
        """Loads custom configuration, unspecified parameters are taken from default configuration.

        Args:
            config (file): file containing configuration json
            debug (boolean): specify to enable debugging features
        """

        with open(os.path.join(os.path.dirname(__file__), "config.json.dist")) as config_file:
            default_config = json.loads(config_file.read())

        custom_config = json.loads(config.read())

        # Merging default and custom configs, for repeating keys, key-value pairs from second dict are taken
        self.nn = {**default_config["neural_net"],
                   **custom_config.get("neural_net", {})}
        self.training = {
            **default_config["training"], **custom_config.get("training", {})}
        self.self_play = {
            **default_config["self_play"], **custom_config.get("self_play", {})}
        self.logging = {**default_config["logging"],
                        **custom_config.get("logging", {})}
        self.storage = {**default_config["storage"],
                        **custom_config.get("storage", {})}
        self.planner = {**default_config["planner"],
                        **custom_config.get("planner", {})}

        self.env = GameEnv(name=self.self_play["game"])
        self.debug = debug


def create_tensorboard_log_dir(logdir, prefix):
    return os.path.join(logdir, prefix, dt.datetime.now().strftime("%d-%mT%H:%M"))


def create_checkpoint_file_name(prefix, game_name, epoch, elo):
    return "_".join([prefix, game_name, '{0:05d}'.format(epoch), str(elo)]) + ".ckpt"


def get_checkpoints_epoch(filename):
    """Get checkpoint epoch from its filename"""

    return int(filename.replace('_', '.').split('.')[-3])


def get_checkpoints_elo(filename):
    """Get checkpoint epoch from its filename"""

    return int(filename.replace('_', '.').split('.')[-2])


def get_newest_ckpt_fname(dirname):
    """Looks for newest file with '.ckpt' extension in dirname."""
    list_of_files = glob.glob(os.path.join(dirname, '*.ckpt'))
    latest_file = max(list_of_files, key=get_checkpoints_epoch)

    return os.path.basename(latest_file)


def get_checkpoints_for_game(dirname, game_name):
    """Looks for files with game_name in filename and '.ckpt' extension in dirname."""
    files = list(filter(os.path.isfile,
                        glob.glob(os.path.join(dirname, '*' + game_name + '*.ckpt'))))
    files.sort(key=lambda x: get_checkpoints_epoch(x))

    return files
