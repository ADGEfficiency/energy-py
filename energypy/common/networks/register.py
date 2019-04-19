""" register for networks """

import logging

from energypy.common.networks.networks import feed_forward_network
from energypy.common.networks.networks import convolutional_network


logger = logging.getLogger(__name__)

network_register = {
    'ff': feed_forward_network,
    'conv': convolutional_network
}


def make_network(network_id, **kwargs):
    logger.info('Making network {}'.format(network_id))

    [logger.debug('{}: {}'.format(k, v)) for k, v in kwargs.items()]

    network = network_register[str(network_id)]

    return network(**kwargs)
