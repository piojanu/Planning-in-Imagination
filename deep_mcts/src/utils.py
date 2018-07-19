import datetime as dt
import glob
import os


def get_newest_ckpt_fname(dirname):
    """Looks for newest file with '.ckpt' extension in dirname."""
    list_of_files = glob.glob(os.path.join(dirname, '*.ckpt'))
    latest_file = max(list_of_files, key=os.path.getctime)

    return os.path.basename(latest_file)


def get_checkpoints_for_game(dirname, game_name):
    """Looks for files with game_name in filename and '.ckpt' extension in dirname."""
    files = list(filter(os.path.isfile,
                        glob.glob(os.path.join(dirname, '*' + game_name + '*.ckpt'))))
    files.sort(key=lambda x: int(x.replace('_', '.').split('.')[-2]))

    return files
