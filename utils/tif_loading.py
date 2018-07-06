"""Utility functions to load data from different sources."""

# Standard imports
import glob
import logging
import os
import random
import math

# Dependency imports
import numpy as np
from osgeo import gdal # pylint: disable=E0401

# Local imports
from visualization import pim

def load_file_paths(data_dir, match_dir_name=None, match_date=None):
    """Load data to numpy array.

    match_dir_name: specific subdirectory to match and load
    match_date: which date to match and load all these date from all subdirectories
    """
    tif_paths = []

    if match_dir_name is None:
        dir_paths = glob.glob(os.path.join(data_dir, '*'))
    else:
        dir_paths = [os.path.join(data_dir, match_dir_name)]

    logging.info('number of directories found %d in %s', len(dir_paths), data_dir)

    for dir_path in dir_paths:

        # We need to extract sensing dates so we can order
        # them and choose only what we need
        sensing_dates = []
        dir_tif_paths = glob.glob(dir_path + '/*.tif')

        for tif_path in dir_tif_paths:
            # Example: S2A_MSIL1C_20171019T093031_N0205_R136_T35VNC_20171019T093214.tif
            # Date at the end is ingestion date

            sensed = os.path.basename(tif_path).split('_')[-1]

            if sensed[0:8] != match_date and match_date is not None:
                continue

            sensing_dates.append(sensed)

        sorted_args = np.argsort(sensing_dates)
        polygon_path = [dir_tif_paths[s_a] for s_a in sorted_args]

        if polygon_path: # If not empty list then append
            tif_paths.append(polygon_path)

    return tif_paths

def read_geotif(tif_paths, nb_time, last_series=True):
    """Load geotif files from directory.

    tif_paths: 2D List where each elemnt is polygon which consist of time series:
               [[t0, t1, t2, ..], [t0, t1, t2, ..]]
    nb_time: how much timeseries to select
    last_series: stretegy of how to select these series. If True, then will select
                 nb_time last series, if False then will select randomly
    """

    out_data = []
    selected_paths = []

    for tif_path in tif_paths:
        sense_dates = {}

        selected_paths.append([])

        # If True then we are dealing with time series
        tif_data = []
        for tif_t_path in tif_path:
            sensed = os.path.basename(tif_t_path).split('_')[-1]

            if sensed not in sense_dates:
                sense_dates[sensed] = False

                tif_t_data = gdal.Open(
                    tif_t_path).ReadAsArray().transpose([1, 2, 0]).astype(np.float32) # pylint: disable=E1101

                # Check if we have any information for date, if have then add it
                if (tif_t_data[tif_t_data.shape[0] // 2,
                               tif_t_data.shape[1] // 2, :] != 0).sum() != 0:

                    tif_data.append(tif_t_data)
                    sense_dates[sensed] = True

                    selected_paths[-1].append(tif_t_path)

        selected_paths[-1] = selected_paths[-1][-nb_time:]

        for key, val in sense_dates.items():
            if not val:
                logging.warning("Found .tif but no useful information for %s"
                                "\nAll polygon paths: %s", key, tif_path)

        if len(tif_data) >= nb_time:

            if last_series:
                tif_data = np.array(tif_data[-nb_time:])
            else:
                rand_idx = random.sample(range(0, len(tif_data)), nb_time)
                tif_data = np.array(tif_data)[rand_idx]

            out_data.append(tif_data)

    logging.debug('number of polygons loaded: %d', len(out_data))
    return out_data

def extract_px_regions(data, class_maps, reg_size):
    """Generate train, valid data and labels

    data: [[polygon_px_0], [polygon_px_1], ...]
    class_maps: How to map found labels in last layer if polygon to actual class used in NN
                e.g. {0:3, 1: 0, 3: 1, 4: 2}
    reg_size: integer which defines resulting size of train data
              e.g. (reg_size x reg_size) where we only classify middle pixel
    """

    assert reg_size % 2 == 1, 'reg_size must be uneven number'

    out_data = []
    out_labels = []

    px_right = math.ceil(reg_size / 2)
    px_left = reg_size // 2

    for polygon in data:

        polygon_data = []
        polygon_labels = []

        similars = [True]
        for serie in polygon[1:, :, :, -1]:
            similars.append(np.array_equal(serie, polygon[0, :, :, -1]))

        if False in similars:
            pim(polygon[:, :, :, -1], cols=6, cmap='gray')
            raise Exception("Labels for single polygon for all time-series "
                            f"must be equal but are: {similars}")

        # Here polygon already consist of time series which is 0th dimension
        # Labels are all same for all time series so we select only 0th time series
        dims_0, dims_1 = np.where(np.isin(polygon[0, :, :, -1], list(class_maps.keys())))

        for dim_0, dim_1 in zip(dims_0, dims_1):
            try:
                pol = polygon[:, dim_0 - px_left: dim_0 + px_right,
                              dim_1 - px_left: dim_1 + px_right, 0:-1]

                if pol.shape[1] == reg_size and pol.shape[2] == reg_size:
                    polygon_data.append(pol)
                    polygon_labels.append(polygon[0, dim_0, dim_1, -1])
                else:
                    logging.debug('Buffer too small: %s %s', pol.shape[0], pol.shape[1])

            except IndexError as err:
                logging.error(err)

        if polygon_labels:
            out_data.append(polygon_data)
            out_labels.append(polygon_labels)

    return out_data, out_labels
