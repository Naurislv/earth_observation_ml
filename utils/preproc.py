"""Utility function to preprocess dataset before NN."""

# Standard imports
import logging

# Dependency imports
import numpy as np
from keras.utils import np_utils

def flatten(data, labels=None):
    """Flatten list.

    data: Feature set. [num, timeseries, height, width, channels]
    data: Label set. [num, nb_class]
    """

    data_flatten = np.concatenate((data))

    if labels is not None:
        labels_flatten = np.concatenate((labels))
        return data_flatten, labels_flatten

    return data_flatten

def normalize(data, norms=None):
    """Normalize data per channel

    data: Feature set. [num, timeseries, height, width, channels]
    """

    data_norm = data.copy()

    def norm_zero_mean(chl, minv, maxv, meanv):
        """Normalize."""

        data_norm[:, :, :, :, chl] = (data_norm[:, :, :, :, chl] - minv - meanv) / (maxv - minv)

    nb_channels = data_norm.shape[-1]

    if norms is None:
        norms = {}

        for channel in range(nb_channels):

            min_val = data_norm[:, :, :, :, channel].min()
            max_val = data_norm[:, :, :, :, channel].max()
            mean_val = data_norm[:, :, :, :, channel].mean()

            norms[channel] = [min_val, max_val, mean_val]
            norm_zero_mean(channel, min_val, max_val, mean_val)

    else:

        for channel, norm in norms.items():

            min_val, max_val, mean_val = norm
            norm_zero_mean(channel, min_val, max_val, mean_val)

    return data_norm

def one_hot(labels_flat, class_maps):
    """Convert list of integer labels to one hot encoded labels.

    labels_flat: List of label numbers e.g. [0, 1, 0, 2, 3, 1, 0 ...]
    class_maps: How to map found labels in last layer if polygon to actual class used in NN
                e.g. {0:3, 1: 0, 3: 1, 4: 2}
    """

    labels_flat_maps = [class_maps[int(lbl)] for lbl in labels_flat]
    labels = np_utils.to_categorical(labels_flat_maps, num_classes=len(class_maps))

    return labels

def add_vegetation(data, nb_b8a, nb_b11, nb_b4):
    """Calculate and add vegetation index as layer.

    data: Feature set. [num, timeseries, height, width, channels]
    b8a: channel number for b8a
    b11: channel number for b11
    b4: channel number for b4
    """

    data_veg = data.copy()

    b_8a = data_veg[:, :, :, :, nb_b8a: nb_b8a + 1]
    b_11 = data_veg[:, :, :, :, nb_b11: nb_b11 + nb_b11]
    b_4 = data_veg[:, :, :, :, nb_b4: nb_b4 + 1]

    ndvi = (b_8a - b_4) / (b_8a + b_4 + 0.001)
    ndwi = (b_8a - b_11) / (b_8a + b_11 + 0.001)

    # When you deal with zero, you get inf., we need correct this
    nan_ndvi = np.isnan(ndvi)
    nan_ndwi = np.isnan(ndwi)

    logging.info('Number of ndvi nan values %d', np.sum(nan_ndvi))
    logging.info('Number of ndwi nan values %d', np.sum(nan_ndwi))

    data_veg = np.concatenate((data_veg, ndvi), axis=-1)
    data_veg = np.concatenate((data_veg, ndwi), axis=-1)

    return data_veg
