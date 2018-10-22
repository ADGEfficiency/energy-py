""" electric grid with market dispatch """
from collections import namedtuple
import heapq as h

import numpy as np
import pandas as pd


Bid = namedtuple('bid', ['price', 'offer', 'dispatch', 'date', 'name'])
Bid.__new__.__defaults__ = (None,) * len(Bid._fields)


def settle_market(bid_stack, demand, date):
    """ finds the cheapest bids to match demand """
    h.heapify(bid_stack)

    dispatch, bids = 0, []
    while dispatch < demand:
        try:
            next_cheapest = h.heappop(bid_stack)
        except IndexError:
            raise ValueError(
                'offers are less than demand of {} MW'.format(demand))
        bid_dispatch = np.min([demand - dispatch, next_cheapest.offer])

        bids.append(
            Bid(price=next_cheapest.price,
                offer=next_cheapest.offer,
                dispatch=bid_dispatch,
                name=next_cheapest.name,
                date=date)
        )

        dispatch += bid_dispatch

    return bids


def test_grid():
    """ settle a simple two bid market """
    market = [
        Bid(price=50, offer=100, name='coal'),
        Bid(price=10, offer=10, name='wind')
    ]

    bids = settle_market(market, demand=50, date='test')

    assert bids[0].dispatch == 10
    assert bids[0].name == 'wind'
    assert bids[1].dispatch == 40
    assert bids[1].name == 'coal'


test_grid()


class Participant():

    def __init__(
            self,
            name,
            prices=(0, 100),
            offers=(0, 50)
    ):
        self.name = name
        self.prices = prices
        self.offers = offers

    def bid(self, observation=None):

        #  random price and offer
        return Bid(
            price=np.random.randint(low=self.prices[0], high=self.prices[1]),
            offer=np.random.randint(low=self.offers[0], high=self.offers[1]),
            name=self.name
        )

market = [
    Participant('wind', prices=(0, 10)),
    Participant('coal', prices=(50, 51))
]

bid_stack = [participant.bid() for participant in market]

dispatch = settle_market(bid_stack, demand=25, date='1')

print(dispatch)

#  could do this in settlemarket
df = pd.concat(
    [pd.DataFrame(bid._asdict(), index=[bid.date]) for bid in dispatch],
    axis=0
)
