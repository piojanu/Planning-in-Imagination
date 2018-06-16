import datetime as dt
import glob
import os


def make_ckpt_fname(game_name, filename="best"):
    """Create ckpt name in format: `filename`_`game_name`_dd-mmTHH:MM:SS.ckpt."""

    # Get rid of extension
    fname = filename.split('.')[0]

    # Get date and time
    datetime = dt.datetime.now().strftime("%d-%mT%H:%M:%S")

    return "_".join([fname, game_name, datetime]) + ".ckpt"


def get_newest_ckpt_fname(dirname):
    """Looks for newest file with '.ckpt' extension in dirname."""
    list_of_files = glob.glob(os.path.join(dirname, '*.ckpt'))
    latest_file = max(list_of_files, key=os.path.getctime)

    return os.path.basename(latest_file)
