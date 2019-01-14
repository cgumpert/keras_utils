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


def resolved_reshaped_features(layer, index):
    """
    Interpret indices of *reshaping* layers in keras.

    Many network architectures employ *reshaping* functionality to transform tensor dimensions suitable for the next
    processing layer (e.g. flattening higher dimensional tensor to be fed to Dense layers, concatenating multiple inputs
    etc.). These layers do not learn features themselves but are merely necessary tools to make the overall network
    architecture work. Since these reshaping operations make it hard to track which *feature* a certain weight
    corresponds to, this function attempts to unroll al the trivial reshaping operations and find the actual feature
    corresponding to a specific weight given by an index and a layer.

    .. note::

       Experimental. Not all types of reshaping layers are supported yet.

    :param layer: keras.layer
    Layer for which the feature corresponding to the layer weight at the given `index` should be looked up.

    :param index: int
    Index of weight in weight matrix of the given layer. The feature related to this weight will be looked up.

    :return: string
    Returns a string describing the actual feature connected to the given weight index. The feature is described in
    terms of the output index of a certain layer. The name, type and output shape of this layer are returned as well.
    """
    def get_input_layer(this_layer, in_index):
        try:
            # noinspection PyProtectedMember
            input_layer = this_layer._inbound_nodes[0].inbound_layers[in_index]
        except IndexError:
            raise RuntimeError("failed to get input layer for layer %r and index %d", this_layer, in_index)
        else:
            return input_layer

    if isinstance(layer, keras.layers.merge.Concatenate):
        axis = layer.axis
        assert index < layer.output_shape[axis], "can't find index %d along axis %d for output shape %r" % (
            index, axis, layer.output_shape)

        # find (relative) index and incoming layer in the list of all input layer which corresponds to the given index
        # in the concatenated layer
        layer_index = 0
        for layer_index, input_layer_shape in enumerate(layer.input_shape):
            if input_layer_shape[axis] > index:
                break
            else:
                index -= input_layer_shape[axis]

        in_layer = get_input_layer(layer, layer_index)
        return resolved_reshaped_features(in_layer, index)
    elif isinstance(layer, keras.layers.Flatten):
        # find original, higher dimensional index for the given flattened index
        in_shape = layer.input_shape
        in_layer = get_input_layer(layer, 0)
        return resolved_reshaped_features(in_layer, np.unravel_index(index, in_shape[1:], 'C'))
    else:
        layer_type = type(layer).__name__.split('.')[-1]
        output_shape = str(layer.output_shape[1:])[1:-1].replace(',', ' x')
        return "{name:s} ({type:s}) @ {index:s} [{shape:s}]".format(name=layer.name,
                                                                    type=layer_type,
                                                                    index=str(index),
                                                                    shape=output_shape)
