"""Entry point to evolving the neural network. Start here."""
from __future__ import print_function

# Standard imports
import logging
import os
import sys

# Dependency imports
from keras import backend as K
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '10' # Disable Tensorflow output

PARENT_PATH = os.path.abspath("..")
if PARENT_PATH not in sys.path:
    sys.path.insert(0, PARENT_PATH)

# pylint: disable=C0413

# Local imports
from utils.dataset import Generator # pylint: disable=E0401, E0611
from evolver import Evolver
from trainer import EvolveTraining

# Set Keras TF backend allow_growth not to consume all GPU memory
K_CONFIG = K.tf.ConfigProto()
K_CONFIG.allow_soft_placement = True
K_CONFIG.gpu_options.allow_growth = True # pylint: disable=E1101
K.set_session(K.tf.Session(config=K_CONFIG))

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='log.txt'
)

def model_func(class_maps, train_dir, valid_dir):
    """Load data and training model function."""

    generator = Generator(
        class_maps=class_maps,
        nb_time=1,
        reg_size=15,
        vegetation_layers=False,
        data_dir=train_dir,
        batch_size=128,
        batch_size_polygons=200
    )

    sample_size = 1000000 # The highest better
    generator.norms_and_weights(sample_size)
    gen = generator.from_files(norm=True, augment=True)

    valid_generator = Generator(
        class_maps={1: 0, 3: 1, 4: 2},
        norms=generator.norms,
        nb_time=1,
        reg_size=15,
        vegetation_layers=False,
        data_dir=valid_dir,
        batch_size=10000,
        batch_size_polygons=10000
    )

    valid_gen = valid_generator.from_files(norm=True, augment=False)
    valid_data, valid_labels = next(valid_gen)

    train = EvolveTraining(gen, len(class_maps), generator.class_weight, valid_data, valid_labels)

    return train.train_and_score

def train_genomes(genomes, train_and_score):
    """Train each genome.

    Args:
        networks (list): Current population of genomes

    """
    logging.info("***train_networks(networks)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(train_and_score)
        pbar.update(1)

    pbar.close()

def get_average_accuracy(genomes):
    """Get the average accuracy for a group of networks/genomes.

    Args:
        networks (list): List of networks/genomes

    Returns:
        float: The average accuracy of a population of networks/genomes.

    """
    total_accuracy = 0

    for genome in genomes:
        total_accuracy += genome.accuracy

    return total_accuracy / len(genomes)

def generate(generations, population, all_possible_genes, train_and_score):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks

    """

    evolver = Evolver(all_possible_genes, opt_for_highest=True)
    genomes = evolver.create_population(population)

    # Evolve the generation.
    for i in range(generations):

        logging.info("***Now in generation %d of %d***", i + 1, generations)

        # print_genomes(genomes)

        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, train_and_score)

        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(genomes)

        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%", average_accuracy * 100)
        logging.info('-' * 80) #-----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:10])

    # save_path = saver.save(sess, '/output/model.ckpt')
    # print("Model saved in file: %s" % save_path)

def print_genomes(genomes):
    """Print a list of genomes.

    Args:
        genomes (list): The population of networks/genomes

    """
    logging.info('-' * 80)

    for genome in genomes:
        genome.print_genome()

class SeArch(object):
    """Define network architecture search space for genetic algorithm."""

    def __init__(self, input_shape, population=3, generations=8):

        """
        population: Number of networks / genomes in each generation. we only
                    need to train the new ones.
        generations: Number of times to evolve the population.

        """

        self.population = population
        self.generations = generations
        self.input_shape = input_shape

        self.nb_layers = None
        self.nb_neurons = None
        self.dropout = None
        self.activation = None

        self.optimizer = None
        self.l2_reg = None
        self.batch_norm = None
        self.l_r = None
        self.epochs = None
        self.batch_size = None
        self.ksize = None
        self.class_weight = None

    def pop_layers(self, val_dict, layer_key, populate_keys):
        """Populate layers with information."""

        for key in populate_keys:
            # replace nb_neurons with 1 unique value for each layer
            # 6th value reserved for dense layer
            for nbl in range(1, max(val_dict[layer_key]) + 1):
                val_dict[f'{key}_{nbl}'] = val_dict[key]
            # remove old value from dict
            val_dict.pop(key)

        return val_dict

    def all_possible_genes(self):
        """Collect all metadata about architecture search and return search space in dictionary."""
        all_possible_genes = {}

        if self.nb_layers is not None:
            all_possible_genes['nb_cnn_layers'] = self.nb_layers
            all_possible_genes['nb_dense_layers'] = self.nb_layers

            pop_cnn_keys = []
            pop_dense_keys = []

            if self.nb_neurons is not None:
                all_possible_genes['__cnn_neurons'] = self.nb_neurons
                all_possible_genes['__dense_neurons'] = self.nb_neurons
                pop_cnn_keys.append('__cnn_neurons')
                pop_dense_keys.append('__dense_neurons')

            if self.dropout is not None:
                all_possible_genes['__cnn_dropout'] = self.dropout
                all_possible_genes['__dense_dropout'] = self.dropout
                pop_cnn_keys.append('__cnn_dropout')
                pop_dense_keys.append('__dense_dropout')

            if self.ksize is not None:
                all_possible_genes['__cnn_ksize'] = self.ksize
                pop_cnn_keys.append('__cnn_ksize')

            self.pop_layers(all_possible_genes, 'nb_cnn_layers', pop_cnn_keys)
            self.pop_layers(all_possible_genes, 'nb_dense_layers', pop_dense_keys)

        if self.activation is not None:
            all_possible_genes['activation_cnn'] = self.activation
            all_possible_genes['activation_dense'] = self.activation

        if self.optimizer is not None:
            all_possible_genes['optimizer'] = self.optimizer

        if self.l2_reg is not None:
            all_possible_genes['l2_reg'] = self.l2_reg

        if self.batch_norm is not None:
            all_possible_genes['batch_norm'] = self.batch_norm

        if self.l_r is not None:
            all_possible_genes['l_r'] = self.l_r

        if self.epochs is not None:
            all_possible_genes['epochs'] = self.epochs

        if self.batch_size is not None:
            all_possible_genes['batch_size'] = self.batch_size

        if self.class_weight is not None:
            all_possible_genes['class_weight'] = self.class_weight

        all_possible_genes['dshape'] = [self.input_shape]

        return all_possible_genes

def main():
    """Evolve a genome."""

    search = SeArch(input_shape=(1, 15, 15, 12), population=70, generations=8)
    search.nb_neurons = [64, 128, 256, 512, 768]
    search.nb_layers = [1, 2, 3, 4]
    search.dropout = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8]
    search.activation = ['relu', 'elu']
    search.l_r = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    search.l2_reg = [1, 0.1, 0.01, 0.001, 0.0001]
    search.batch_norm = [True, False]
    search.epochs = [10, 15, 20, 30, 40, 50, 60, 70]
    search.ksize = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    search.class_weight = [True, False]
    # search.batch_size = [8, 16, 32, 64, 128]

    all_possible_genes = search.all_possible_genes()

    class_maps = {1: 0, 3: 1, 4: 2}
    train_and_score = model_func(class_maps, '../../data/_train/', '../../data/_valid/')

    logging.info("Evolving for %d generations with population size = %d",
                 search.generations, search.population)
    generate(search.generations, search.population, all_possible_genes, train_and_score)

if __name__ == '__main__':
    main()
