import io
import pkg_resources

import pandas as pd


def load_dataset(dataset, name):
    """ TODO only gets example data """

    path = 'experiments/datasets/example/{}.csv'.format(name)

    data = pkg_resources.resource_string('energypy', path)

    csv = pd.read_csv(
        io.BytesIO(data), index_col=0, parse_dates=True
    )

    return csv
