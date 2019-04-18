"""PlaNet hyper-params random search optimization.

Config (YAML) format:
```
<param name>: [<base>, [<exp lower bound>, <exp higher bound>]]
<param name>: <constant value>
```

Config example:
```
image_loss_scale: [10, [-3, 1]]
divergence_scale: [10, [-4, 0]]
max_steps: 1e6
```

Exponential is sampled uniformly and param is calculated as: base ** exponential.
"""

import argparse
import os

import numpy as np
import ruamel.yaml as yaml

from planet import tools
from planet.scripts.train import main


def construct_suffix(params):
    """Constructs suffix from params' keys and values.

    Args:
        params (tools.AttrDict): PlaNet hyper-params.
    """

    suffix = ""
    for param, value in params.items():
        if "tasks" in param or "max_steps" in param:
            continue

        try:
            suffix += "_{}={:.3E}".format(param, value)
        except ValueError:
            suffix += "_{}={}".format(param, value)

    return suffix


def hyperopt(args, max_iters=None):
    """PlaNet hyper-params random search optimization.

    Note: `args` include `params` which includes bases, low and high exponential search ranges for
          each hyper-param.

    Args:
        args (Namespace): Command line arguments used to construct PlaNet train params.
        max_iters (int): Hyper-param optimisation iterations limit. If `None` then unlimited.
    """

    # Copy original values
    logdir = args.logdir
    params = args.params.copy()

    step = 0
    while max_iters is None or step < max_iters:
        # Reset params
        args.params = params.copy()

        # Sample hyper-params
        with args.params.unlocked:
            for param, value in params.items():
                try:
                    a, b = value
                except (TypeError, ValueError):
                    continue

                try:
                    base, bounds = a, b
                    exp = np.random.uniform(*bounds)
                    args.params[param] = base ** exp
                except TypeError:
                    low, high = a, b
                    value = np.random.randint(low, high)
                    args.params[param] = value

        # Update logdir path
        args.logdir = os.path.join(logdir, "run" + construct_suffix(args.params))

        # Run training
        main(args)

        step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlaNet hyper-params random search optimization.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--iters', type=int, default=None,
                        help="Number of hyper-param. optimisation iterations.")
    parser.add_argument('--logdir', type=str,
                        help="Where to save logs.")
    parser.add_argument('--num_runs', type=int, default=3,
                        help="How many times to run every sampled parameters.")
    parser.add_argument('--config', type=str, default='default',
                        help='Select a configuration function from scripts/configs.py.')
    parser.add_argument('--params', type=str, default='{}',
                        help='YAML formatted dictionary to be used by the config.')
    parser.add_argument('--ping_every', type=int, default=0,
                        help='Used to prevent conflicts between multiple workers; 0 to disable.')
    parser.add_argument('--resume_runs', type=bool, default=False,
                        help='Whether to resume unfinished runs in the log directory.')
    args = parser.parse_args()

    args.params = tools.AttrDict(yaml.safe_load(args.params.replace('#', ',')))
    args.logdir = args.logdir and os.path.expanduser(args.logdir)
    hyperopt(args, max_iters=args.iters)
