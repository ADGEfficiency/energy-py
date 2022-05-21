"""
   query: Query Tensor of shape (B, T, dim).
   value: Value Tensor of shape (B, S, dim).
   key Optional key Tensor of shape (B, S, dim). If not given, will use value for both key and value, which is the most common case.

   attention_mask: a boolean mask of shape (B, T, S), that prevents attention to certain positions. The boolean mask specifies which query elements can attend to which key elements, 1 indicates attention and 0 indicates no attention. Broadcasting can happen for the missing batch dimensions and the head dimension.

    return_attention_scores: A boolean to indicate whether the output should be attention output if True, or (attention_output, attention_scores) if False. Defaults to False.

attention_output    The result of the computation, of shape (B, T, E), where T is for target sequence shapes and E is the query input last dimension if output_shape is None. Otherwise, the multi-head outputs are project to the shape specified by output_shape. 

B = batch
T = target sequence length
S = source sequence length

we are doing self-attention so T = S
"""
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from energypy.networks import attention


def test_attention():
    """
    just check we can run it
    """
    n_samples = 2
    sequence_length = 2
    n_features = 3

    query = np.random.random((n_samples, sequence_length, n_features))
    mask = np.random.random((n_samples, sequence_length, sequence_length)).astype(bool)
    inp, out = attention((sequence_length, n_features), 1)
    mdl = keras.Model(inputs=inp, outputs=out)

    #  (2, 1)
    print(mdl([query, mask]).shape)


test_attention()
