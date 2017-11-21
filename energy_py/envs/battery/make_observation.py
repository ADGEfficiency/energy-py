"""
This script creates the 'observation.csv' from the 'state.csv'

We add in some additional datetime features to help our model learn.
"""
import numpy as np
import pandas as pd

#  read in the raw data
print('reading in state.csv')
state = pd.read_csv('state.csv', index_col=0, header=0, parse_dates=True)
print('read state.csv')

#  checking that we only have continuous variables in our state csv
#  will integrate dummy variables in state eventually

agent_horizion = 5 #  12 5 min periods per hour

#  lag out the price column
price = state.loc[:, 'C_electricity_price_[$/MWh]']
forecast = pd.concat([price.shift(-i) for i in range(agent_horizion)], axis=1).dropna()
forecast.index = pd.to_datetime(forecast.index)
dfs = [forecast]

print('finishing making horizions')

def make_datetime_features(index):
    #  make some datetime features
    print('making date time features')
    month = [index[i].month for i in range(len(index))]
    day = [index[i].day for i in range(len(index))]
    hour = [index[i].hour for i in range(len(index))]
    minute = [index[i].minute for i in range(len(index))]
    weekday = [index[i].weekday() for i in range(len(index))]

    #  turn the datetime features into dummies
    features = [month, day, hour, minute, weekday]
    feature_names = ['month', 'day', 'hour', 'minute', 'weekday']

    #  loop over the created dummies
    dummies = []
    for feature, name in zip(features, feature_names):
        dummy = pd.get_dummies(feature)
        dummy.columns = ['D_' + name + '_' +str(col) for col in dummy.columns]
        dummy.index = index
        dummies.append(dummy)
    return dummies

make_dt_features = True
if make_dt_features:
    dfs.extend(make_datetime_features(forecast.index))

observation = pd.concat(dfs, axis=1)
observation.dropna(axis=0, inplace=True)
observation['counter'] = np.arange(observation.shape[0])
print(observation.head(5))

print('saving observation.csv')
observation.to_csv('observation.csv')
