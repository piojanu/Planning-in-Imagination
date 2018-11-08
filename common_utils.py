import json
import os
import tensorflow as tf

from keras.backend.tensorflow_backend import set_session


def get_configs(config_path):
    """Loads default and custom configs
        Args:
            config_path (str): Path to .json file with custom configuration

        Return:
            dict: Default config
            dict: Custom config
    """

    with open(os.path.join(os.path.dirname(__file__), "config.json.dist")) as config_file:
        default_config = json.loads(config_file.read())

    if os.path.exists(config_path):
        with open(config_path) as custom_config_file:
            custom_config = json.loads(custom_config_file.read())
    else:
        custom_config = {}
    return default_config, custom_config


def limit_gpu_memory_usage():
    """This function makes that we don't allocate more graphics memory than we need.
       For TensorFlow, we need to set `alow_growth` flag to True.
       For PyTorch, this is the default behavior.

    """

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))


def create_directory(dirname):
    """Create directory recursively, if it doesn't exit

    Args:
        dirname (str): Name of directory (path, e.g. "path/to/dir/")
    """

    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)


def force_cpu():
    """Force using CPU"""

    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def mute_tf_logs_if_needed():
    if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
