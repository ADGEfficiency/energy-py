import io
from os.path import join
import pkg_resources

# import pandas as pd


def load_dataset(dataset, name):
    """ load example dataset or load from user supplied path """
    if dataset == 'example':
        path = 'experiments/datasets/example/{}.csv'.format(name)
        data = pkg_resources.resource_string('energypy', path)

        return pd.read_csv(
            io.BytesIO(data), index_col=0, parse_dates=True
        )

    else:
        return pd.read_csv(
            join(dataset, name + '.csv'),
            index_col=0,
            parse_dates=True
        )
