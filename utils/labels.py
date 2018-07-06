"""Download labels from postgis DB and preprocess them for neural network."""

# Standard imports
import logging

# Dependency imports
import numpy as np

# Local imports
from postgis_client import GisDB


class Labels(GisDB):
    """Prepare labels for training."""

    def __init__(self):
        GisDB.__init__(self)

        self._species = None
        self._specie_threshold = 1000

    @property
    def specie_threshold(self):
        """Return current threshold value."""
        return self._specie_threshold

    @specie_threshold.setter
    def specie_threshold(self, specie_threshold):
        "If specie class is presented less times than this number then convert to 'others' class"

        if specie_threshold != self._specie_threshold:
            self._specie_threshold = specie_threshold
            self._species = None

    @property
    def species(self):
        """Fill NaN vallues for given columns."""

        if self._species is None:
            print('AAA')

            data = self.data.copy()
            logging.info('Input data shape: %s', data.shape)

            data['s2'].fillna(-1, inplace=True)
            data['s2'] = data['s2'].astype(int)
            data['s3'].fillna(-1, inplace=True)
            data['s3'] = data['s3'].astype(int)

            data['k2'].fillna(0.0, inplace=True)
            data['k3'].fillna(0.0, inplace=True)

            class_names = [i for i in self.nb_class if self.nb_class[i] >= self._specie_threshold]
            class_ids = [self.class_maps_inv[cls] for cls in class_names]

            logging.info('Selecting following class names: %s', class_names)
            logging.info('Selecting following class ids: %s', class_ids)

            data.loc[~data.s1.isin(class_ids + [-1]), 's1'] = 99
            data.loc[~data.s2.isin(class_ids + [-1]), 's2'] = 99
            data.loc[~data.s3.isin(class_ids + [-1]), 's3'] = 99

            class_ids = sorted(class_ids + [99])
            logging.info("Number of classes including 'others': %s", len(class_ids))

            # For classifying species we don't same class in different part, so let's sum them
            for p_0, p_1 in [('1', '2'), ('1', '3'), ('2', '3')]:

                s12 = data[f's{p_0}'] == data[f's{p_1}']
                data.loc[s12, f'k{p_0}'] = data[f'k{p_0}'][s12] + data[f'k{p_1}'][s12]
                data.loc[s12, f'k{p_1}'] = 0.0
                data.loc[s12, f's{p_1}'] = -1

            self._species = np.zeros((data.shape[0], len(class_ids)), dtype=np.float32) # pylint: disable=E1101

            for idx, row in data[['s1', 's2', 's3', 'k1', 'k2', 'k3']].iterrows():

                s1_idx = class_ids.index(row['s1'])
                self._species[idx, s1_idx] = row['k1']

                try:
                    s2_idx = class_ids.index(row['s2'])
                    self._species[idx, s2_idx] = row['k2']
                except ValueError:
                    pass

                try:
                    s3_idx = class_ids.index(row['s3'])
                    self._species[idx, s3_idx] = row['k3']
                except ValueError:
                    pass

                # TODO: This is not the best way how to deal with absence of all presented tree species in data
                if sum(self._species[idx]) < 0.999:
                    self._species[idx, -1] = round(self._species[idx, -1] +
                                                   1 - sum(self._species[idx]), 1)

        return self._species

# TODO: Dataset tests

# d_f_fillna[(d_f_fillna['k1'] + d_f_fillna['k2'] + d_f_fillna['k3']) > 1]
# d_f_fillna[['s1', 's2']].groupby(['s1', 's2']).size().unstack(fill_value=0).head(5)
# d_f_fillna[['s1', 's3']].groupby(['s1', 's3']).size().unstack(fill_value=0).head(5)
# d_f_fillna[['s2', 's3']].groupby(['s2', 's3']).size().unstack(fill_value=0).head(5)

# There shouldn't be no same values between s, h, a columns
# mask_0 = ((d_f_fillna['s1'] == d_f_fillna['s2']) &
#         (d_f_fillna['h1'] == d_f_fillna['h2']) &
#         (d_f_fillna['a1'] == d_f_fillna['a2']))

# mask_1 = ((d_f_fillna['s1'] == d_f_fillna['s3']) &
#           (d_f_fillna['h1'] == d_f_fillna['h3']) &
#           (d_f_fillna['a1'] == d_f_fillna['a3']))

# mask_2 = ((d_f_fillna['s2'] == d_f_fillna['s3']) &
#           (d_f_fillna['h2'] == d_f_fillna['h3']) &
#           (d_f_fillna['a2'] == d_f_fillna['a3']))

# print(d_f_fillna[mask_0].shape)
# print(d_f_fillna[mask_1].shape)
# print(d_f_fillna[mask_2].shape)


if __name__ == "__main__":

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO
    )

    LABELS = Labels()
    LABELS.specie_threshold = 500
