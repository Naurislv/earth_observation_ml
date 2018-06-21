"""Makes predictions for satelite map data to classify stuff."""

# Standard imports
import os
import sys
import json

# Dependency imports
import tensorflow as tf
import numpy as np
from keras.models import model_from_json
import keras.backend.tensorflow_backend as K

# Local imports
from utils.dataset import Loader

# Set Keras TF backend allow_growth not to consume all GPU memory
K_CONFIG = K.tf.ConfigProto()
K_CONFIG.allow_soft_placement = True
K_CONFIG.gpu_options.allow_growth = True # pylint: disable=E1101
K.set_session(K.tf.Session(config=K_CONFIG))

class Classifier(object):
    """Satellite image classifier for inference."""

    def __init__(self, model_dir=None):

        self.graph = None

        self.model = self.load_model(os.path.join(model_dir, 'model'))

        with open(os.path.join(model_dir, 'stats.json')) as json_data:
            stats = json.load(json_data)

         # Automatically key is set to string we'd like to use it as integer
        self.norms = {int(k): v for k, v in stats['norms'].items()}
        self.class_maps = {int(k): v for k, v in stats['maps'].items()}
        self.inv_class_maps = {v: k for k, v in self.class_maps.items()}

    def load_model(self, model_path):
        """Load Keras model architecture and weights."""

        j_path = model_path + '.json'
        print(j_path)
        with open(j_path, 'r') as jfile:
            model = model_from_json(jfile.read())
        model.compile("adam", "categorical_crossentropy")
        print('Model loaded:', j_path)

        self.graph = tf.get_default_graph()

        w_path = model_path + '.h5'
        model.load_weights(w_path)
        print('Weights loaded:', w_path)

        return model

    def predict_shp(self, shp_nr):
        """Load file from local directory with shp nr."""

        dataset = Loader(
            norms=self.norms,
            class_maps=self.class_maps,
            data_dir='../data/final_latest/',
            nb_split_polygons=0,
            nb_time_series=1,
            reg_size=15,
            match_shape_nr=str(shp_nr)
        )

        preds = self.model.predict(dataset.train_data)

        pred_region = dataset.input_data[0][0, :, :, -1]
        pred_region[pred_region > 0] = np.argmax(preds, axis=1)

        return pred_region

    def predict(self, geotif):
        """Make prediction and return label."""

        # for chl, norm in self.norms.items():
        #     min_val, max_val, mean_val = norm

        #    geotif[:, :, :, :, chl] = ((geotif[:, :, :, :, chl] - min_val - mean_val) /
        #                                (max_val - min_val))

        softmax_preds = self.model.predict(geotif)
        arg_max = np.argmax(softmax_preds, axis=1)

        preds = []
        confidences = []
        for i, arg_idx in enumerate(arg_max):
            preds.append(self.inv_class_maps[arg_idx])
            confidences.append(softmax_preds[i][arg_idx])

        return preds, confidences

if __name__ == "__main__":

    SHP_NR = sys.argv[1]

    CLFR = Classifier('inference_model')
    PREDS, CONFIDENCE = CLFR.predict_shp(SHP_NR)

    print(np.argmax(PREDS, axis=1))
    print(np.argmax(CONFIDENCE, axis=1))
