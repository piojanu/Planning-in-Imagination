import datetime as dt
import glob
import os


def create_tensorboard_log_dir(logdir, prefix):
    return os.path.join(logdir, prefix, dt.datetime.now().strftime("%d-%mT%H:%M"))


def get_checkpoints_epoch(filename):
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
    files.sort(key=lambda x: int(x.replace('_', '.').split('.')[-2]))

    return files
