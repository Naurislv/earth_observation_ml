"""
Generic setup of the data sources and the model training. 

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on 
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""

# Standard imports
import logging
import sys
import os

PARENT_PATH = os.path.abspath("..")
if PARENT_PATH not in sys.path:
    sys.path.insert(0, PARENT_PATH)

# Dependency imports
from keras import backend as K # pylint: disable=C0413

# Local imports
from models.keras_net import train_model # pylint: disable=E0401, C0413

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO#,
    # filename='log.txt'
)

class EvolveTraining(object):
    """Training class for genetics algorithm."""

    def __init__(self, generator, num_class, class_weight, valid_data, valid_labels):

        self.generator = generator
        self.num_class = num_class
        self.class_weight = class_weight
        self.valid_data = valid_data
        self.valid_labels = valid_labels

    def train_and_score(self, genome):
        """Train the model, return test loss.

        Args:
            network (dict): the parameters of the network

        """
        print(genome.geneparam)
        logging.info("Compiling Keras model. Will train on %s epochs", genome.geneparam['epochs'])

        if genome.geneparam['class_weight']:
            set_class_weight = self.class_weight
        else:
            set_class_weight = None

        _, keras_model = train_model(
            self.generator,
            self.num_class,
            genome.geneparam,
            set_class_weight,
            200,
            self.valid_data,
            self.valid_labels,
            verbose=0
        )

        score = keras_model.evaluate(self.valid_data, self.valid_labels, verbose=0)
        logging.info('Test loss: %s, Test accuracy: %s', score[0], score[1])

        K.clear_session()
        # we do not care about keeping any of this in memory -
        # we just need to know the final scores and the architecture

        return score[1] # 0 is loss, 1 is accuracy.
