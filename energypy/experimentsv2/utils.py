from io import BytesIO
import json
import os
import pkg_resources

import pandas as pd


def read_logs(log_file_path):
    with open(log_file_path) as f:
        logs = f.read().splitlines()

    return [json.loads(log) for log in logs]


def load_dataset(dataset, name):
    """ load example dataset or load from user supplied path """
    if dataset == 'example':
        data = pkg_resources.resource_string(
            'energypy',
            'experiments/datasets/example/{}.csv'.format(name)
        )

        return pd.read_csv(
            BytesIO(data), index_col=0, parse_dates=True
        )

    else:
        return pd.read_csv(
            os.path.join(dataset, name + '.csv'), index_col=0, parse_dates=True
        )


def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def dump_config(cfg, logger):
    for k, v in cfg.items():
        logger.info(json.dumps({k: v}))

