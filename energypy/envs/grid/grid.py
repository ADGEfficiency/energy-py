""" electric grid with market dispatch """
from collections import namedtuple
import heapq as h

import numpy as np


Bid = namedtuple('bid', ['price', 'offer', 'dispatch', 'name'])


def settle_market(market, demand):
    """ finds the cheapest bids to match demand """
    h.heapify(market)

    dispatch, bids = 0, []
    while dispatch < demand:
        next_cheapest = h.heappop(market)
        bid_dispatch = np.min([demand - dispatch, next_cheapest.offer])

        bids.append(
            Bid(next_cheapest.price,
                next_cheapest.offer,
                bid_dispatch,
                next_cheapest.name)
        )

        dispatch += bid_dispatch

    return bids


def test_grid():
    """ settle a simple two bid market """
    market = [
        Bid(50, 100, None, 'coal'),
        Bid(10, 10, None, 'wind')
    ]

    bids = settle_market(market, demand=50)

    assert bids[0].dispatch == 10
    assert bids[0].name == 'wind'
    assert bids[1].dispatch == 40
    assert bids[1].name == 'coal'


test_grid()
