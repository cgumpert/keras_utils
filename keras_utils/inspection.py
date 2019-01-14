import keras
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def interpret_conv1d_weights(weights, alphabet):
    """
    Interpret weights of 1D convolution filters in NLP tasks.

    In order to understand what features a 1D convolution layer has learned in the NLP domain one must map the various
    kernel weights to the corresponding input tokens (e.g. words or characters) and their relative position in the
    kernel window. This function performs the mapping and returns the convolution kernels in a human-understandable form
    as dataframes. These dataframes list the for each token and relative position in the kernel window the corresponding
    weight.

    :param weights: numpy array of shape (kernel_size x alphabet_size x n_filters)
    This is the weight matrix for the convolution kernels.

    :param alphabet: List-like of type string with length alphabet_size
    List of tokens or characters which serve as input to the 1D convolution kernels. These input tokens are assumed to
    be one-hot encoded where the position in `alphabet` corresponds to the index of the 1 in the corresponding one-hot
    encoded representation.

    :return: List of pandas.DataFrame
    For each filter a dataframe with the weight for each input token at a given position in the kernel window is
    returned.
    """
    assert len(weights.shape) == 3, "expect input weights of dimension 3 (got %d)" % len(weights.shape)
    filters = []
    window, n_tokens, n_filters = weights.shape
    assert len(alphabet) == n_tokens, "alphabet size does not match token dimension of kernel weights" \
        " (got %d expected %d)" % (len(alphabet), n_tokens)
    for f_id in range(n_filters):
        position_weights = []
        for pos_id in range(window):
            this_weights = pd.Series(index=[alphabet[a_id] for a_id in range(n_tokens)],
                                     data=[weights[pos_id, a_id, f_id] for a_id in range(n_tokens)],
                                     name='weight_%d' % pos_id)
            this_weights.index.name = 'token_%d' % pos_id
            position_weights.append(this_weights)
        this_filter = pd.concat([w_df
                                .sort_values(ascending=False)
                                .reset_index()
                                 for w_df in position_weights], axis=1)
        filters.append(this_filter)

    return filters
