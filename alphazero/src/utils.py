import datetime as dt
import glob
import os
import tensorflow as tf


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


def create_tensorboard_log_dir(logdir, prefix):
    return os.path.join(logdir, prefix, dt.datetime.now().strftime("%d-%mT%H:%M"))


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
