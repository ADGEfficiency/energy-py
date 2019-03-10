from io import BytesIO
from os.path import join
import pkg_resources

import pandas as pd


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
            join(dataset, name + '.csv'), index_col=0, parse_dates=True
        )
