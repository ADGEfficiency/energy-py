"""
This script was used to generate the state CSV for the battery environment
Running it again should be done with caution - the initial raw data was overwritten!

The raw data for the 'electricity_price' is the historical price of
electricity in the South Australian NEMWEB market.

The data was provided by AEMO here - http://www.nemweb.com.au/REPORTS/

The 'electricity_demand' is modelled at a constant 10 MW.

It is envisoned that the user will input their own electricity price and
demand data.  Note that this environment works on a 5 minute frequency.

We add in some additional datetime features to help our model learn.
"""

import pandas as pd

#  read in the raw data
state = pd.read_csv('state.csv', index_col=0, header=0, parse_dates=True)

#  make some datetime features
index = state.index
state.loc[:, 'month'] = [index[i].month for i in range(len(index))]
state.loc[:, 'day'] = [index[i].day for i in range(len(index))]
state.loc[:, 'hour'] = [index[i].hour for i in range(len(index))]
state.loc[:, 'minute'] = [index[i].minute for i in range(len(index))]
state.loc[:, 'weekday'] = [index[i].weekday() for i in range(len(index))]

#  save the state CSV
state.to_csv('state.csv')
