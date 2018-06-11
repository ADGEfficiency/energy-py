"""
A collection of functions to run experiments.

Module contains:
    make_expt_parser - parses command line arguments for experiments
    make_paths - creates a dictionary of paths
    run_config_expt - runs an experiment using a config file
    experiment - runs a single reinforcment learning experiment
    Runner - class to save environment data & TensorBoard
"""

import logging
import logging.config
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import energy_py
from energy_py.experiments.datasets import get_dataset_path


logger = logging.getLogger(__name__)





