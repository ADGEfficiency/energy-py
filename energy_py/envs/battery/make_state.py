"""
This script creates the 'state.csv' from the 'raw_state.csv'

Should only be run once so not hyperoptimized code.

Main point is the creation of dummy variables from the time series info.

Assumed that all other variables are continuous.

The raw data for the 'electricity_price' is the historical price of
electricity in the South Australian NEMWEB market.

The data was provided by AEMO here - http://www.nemweb.com.au/REPORTS/

The 'electricity_demand' is modelled at a constant 10 MW.

It is envisoned that the user will input their own electricity price and
demand data.  Note that this environment works on a 5 minute frequency.

We add in some additional datetime features to help our model learn.
"""

import 
import pandas as pd

#  read in the raw data
print('reading in raw_state.csv')
state = pd.read_csv('raw_state.csv', index_col=0, header=0, parse_dates=True)

#  checking that we only have continuous variables in our state csv
#  will integrate dummy variables in raw_state eventually

agent_horizion = 8 * 12  #  12 5 min periods per hour 
dfs = []
for col in state.columns:
    assert str(col[:2]) == 'C_'

    fc = pd.concat([state.loc[:,col].shift(-i) for i in
                    range(agent_horizion)],axis=1)
    dfs.append(fc)

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

state = pd.concat(dfs, axis=1)
state.dropna(axis=0, inplace=True)

print(state.head(5))
#  save the state CSV
print('saving state.csv')
state.to_csv('state.csv')
