"""Sentinels data preparation for training NN."""

# Standard imports
from math import ceil
import random
import sys
import os

# Dependency imports
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Local imports
from data_loading import load_file_paths, read_geotif, extract_px_regions # pylint: disable=C0413
from preproc import flatten, normalize, one_hot, add_vegetation # pylint: disable=C0413

class Augmentation(object):
    """Rotation, flip data augmentation with random function. Made for specific use-case."""

    def rot90(self, data):
        """Rotate by 90 degrees."""
        return np.rot90(data, 1, axes=(2, 3))

    def rot180(self, data):
        """Rotate by 180 degrees."""
        return np.rot90(data, 2, axes=(2, 3))

    def rot270(self, data):
        """Rotate by 270 degrees."""
        return np.rot90(data, 3, axes=(2, 3))

    def flip(self, data):
        """Flip."""
        return np.flip(data, axis=3)

    def flip_rot90(self, data):
        """Flip rotate by 90 degrees."""
        return self.rot180(self.flip(data))

    def orig(self, data):
        """Original data."""
        return data

    def random_augm(self, data):
        """Randomly select one of transformations."""

        data_augm = data.copy() # make sure that original data is not affected
        augm_func = random.choice([self.rot90, self.rot180, self.rot270,
                                   self.flip, self.flip_rot90, self.orig])

        return augm_func(data_augm)

class Generator(Augmentation):
    """Abstraction DataLoader class to create generator."""

    def __init__(self, data_dir, class_maps, batch_size, batch_size_polygons,
                 norms=None, nb_time=1, reg_size=1, vegetation_layers=False):
        Augmentation.__init__(self)

        self.class_maps = class_maps
        self.norms = norms
        self.reg_size = reg_size
        self.nb_time = nb_time
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.batch_size_polygons = batch_size_polygons

        self.vegetation_layers = vegetation_layers

        self.class_weight = {}

    def from_files(self, norm=True, augment=False, match_dir_name=None):
        """Generator for data reading and augmentation.

        batch_size: Feature returning batch size
        batch_size_polygons: How much polygons read in once

        ** All other parameters are DataLoader related. **
        """

        tif_paths = np.array(load_file_paths(self.data_dir, match_dir_name))
        nb_polygons = len(tif_paths)

        bsp = self.batch_size_polygons
        if self.batch_size_polygons > nb_polygons:
            bsp = nb_polygons

        while True:

            # Choose batch_size_polygons polygons to read by randomly selecting indexes
            poly_idx = np.random.choice(nb_polygons, size=bsp, replace=False)

            # Read polygons as arrays
            poly_data = read_geotif(tif_paths[poly_idx], self.nb_time, last_series=True)
            # Extract regions from polygons
            data, labels = extract_px_regions(poly_data, self.class_maps, self.reg_size)

            # Add preprocessing, after this step feature are ready to be used
            data, labels = self.preproc(data, labels, norm)
            fshape = data.shape

            b_s = self.batch_size
            if self.batch_size > fshape[0]:
                b_s = fshape[0]

            # Iterate through read feature fshape[0] // batch_size times
            for _ in range(ceil(fshape[0] / b_s)):

                # Choose batch_size features to read by randomly selecting indexes
                feature_idx = np.random.choice(fshape[0], size=b_s, replace=False)

                # Augment train_dataset with rotations
                if augment:
                    batch_features = self.random_augm(data[feature_idx])
                else:
                    batch_features = data[feature_idx]
                batch_labels = labels[feature_idx]

                yield batch_features, batch_labels

    def preproc(self, data, labels, norm=True):
        """Apply different preprocessing function on data. Prepare data for NN."""

        data_proc = data.copy()
        labels_proc = labels.copy()

        data_proc = flatten(data_proc)

        if self.vegetation_layers:
            data_proc = add_vegetation(data_proc, 5, 6, 10)

        if norm:
            data_proc = normalize(data_proc, self.norms)

        labels_proc = flatten(labels_proc)
        labels_proc = one_hot(labels_proc, self.class_maps)

        return data_proc, labels_proc

    def norms_and_weights(self, sample_size):
        """Calculate norms and weights on larger sample size.."""

        def n_unique(arr, val_dict):
            """Count unique values"""
            bincount = np.bincount(arr)
            b_ii = np.nonzero(bincount)[0]

            for val, count in zip(b_ii, bincount[b_ii]):

                if val not in val_dict:
                    val_dict[val] = count
                else:
                    val_dict[val] += count

            return val_dict

        gen = self.from_files(norm=False, augment=True)

        self.norms = {}
        unique_vals = {}
        unique_vals['total'] = 0
        iter_n = sample_size // self.batch_size

        for _ in range(iter_n):
            x_data, y_data = next(gen)

            unique_vals = n_unique(np.argmax(y_data, axis=1), unique_vals)
            unique_vals['total'] += x_data.shape[0]

            n_channels = x_data.shape[-1]
            for i in range(n_channels):

                min_val = x_data[:, :, :, :, i].min()
                max_val = x_data[:, :, :, :, i].max()
                mean_val = x_data[:, :, :, :, i].mean()

                if i not in self.norms:
                    self.norms[i] = [min_val, max_val, mean_val]
                else:

                    if min_val < self.norms[i][0]:
                        self.norms[i][0] = min_val
                    if max_val > self.norms[i][1]:
                        self.norms[i][1] = max_val

                    self.norms[i][2] += mean_val

        # Calculate mean for all itterations
        for key in self.norms:
            self.norms[key][2] = self.norms[key][2] / iter_n

        # Calculcate class weights
        for key, val in unique_vals.items():

            if key != 'total':
                self.class_weight[key] = unique_vals['total'] / val


class Loader(object):
    """Load data for training."""

    def __init__(self):
        raise NotImplementedError
