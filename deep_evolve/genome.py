"""The genome to be evolved."""

# Standard imports
import random
import logging
import hashlib
import copy

def cnn_output(input_size, kernels):
    """Calculate output size of CNN and if not negative then will return True."""

    lsize = input_size

    for krnl in kernels:

        lsize = lsize - krnl + 1

        if lsize < 1:
            return False

    return True

class Genome(object):
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """

    def __init__(self, all_possible_genes=None, geneparam={},
                 u_id=0, mom_id=0, dad_id=0, gen=0):
        """Initialize a genome.

        Args:
            all_possible_genes (dict): Parameters for the genome, includes:
                gene_nb_neurons_i (list): [64, 128, 256]      for (i=1,...,6)
                gene_nb_layers (list):  [1, 2, 3, 4]
                gene_activation (list): ['relu', 'elu']
                gene_optimizer (list):  ['rmsprop', 'adam']
        """
        self.accuracy = 0.0
        self.all_possible_genes = all_possible_genes
        self.geneparam = geneparam # (dict): represents actual genome parameters
        self.u_id = u_id
        self.parents = [mom_id, dad_id]
        self.generation = gen

        # hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()

    def update_hash(self):
        """
        Refesh each genome's unique hash - needs to run after any genome changes.
        """

        self.validate_generation()

        genh = ""
        genp = self.geneparam.copy()

        del_keys = []
        for key in genp:
            if '__dense' in key:
                l_dense = int(key.split('_')[-1])
                if l_dense > genp['nb_dense_layers']:
                    del_keys.append(key)
            if '__cnn' in key:
                l_dense = int(key.split('_')[-1])
                if l_dense > genp['nb_cnn_layers']:
                    del_keys.append(key)

        # Delete all neuron information about layers we don't have for this genom
        for key in del_keys:
            del genp[key]

        for key in sorted(genp):
            genh += str(genp[key])

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()
        self.accuracy = 0.0

    def set_genes_random(self):
        """Create a random genome."""
        self.parents = [0, 0]

        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])

        self.update_hash()

    def mutate_one_gene(self):
        """Randomly mutate one gene in the genome.

        Args:
            network (dict): The genome parameters to mutate

        Returns:
            (Genome): A randomly mutated genome object

        """

        all_genes = list(self.all_possible_genes.keys())
        for _ in range(len(all_genes) * 3):
            # Which gene shall we mutate? Choose one of N possible keys/genes.
            gene_to_mutate = random.choice(all_genes)

            # And then let's mutate one of the genes.
            # Make sure that this actually creates mutation
            current_value = self.geneparam[gene_to_mutate]
            possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])

            if len(possible_choices) > 1:
                possible_choices.remove(current_value)
                self.geneparam[gene_to_mutate] = random.choice(possible_choices)
                self.update_hash()

                break
            else:
                logging.warning(f"Only one possible value for {gene_to_mutate}, cannot mutate")

    def validate_generation(self):
        """There is possibility to set genes so that it's not
        possible to use generation for specific purpose."""

        cnn_valid = False
        ksizes = [val for key, val in sorted(self.geneparam.items()) if '__cnn_ksize' in key]

        # Suggest new value.
        while True:

            cnn_valid = cnn_output(
                self.geneparam['dshape'][1],
                ksizes
            )
            if cnn_valid:
                break

            # Pick random layer to decrease kernel size
            rand_idx = random.randint(0, len(ksizes) - 1)

            if ksizes[rand_idx] > 1:
                ksizes[rand_idx] -= 1

        set_idx = 0
        for key in sorted(self.geneparam):
            if '__cnn_ksize' in key:
                self.geneparam[key] = ksizes[set_idx]
                set_idx += 1

    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased"""

        self.generation = generation
        logging.info("Setting Generation to %d", self.generation)

    # def set_genes_to(self, geneparam, mom_ID, dad_ID):
    #     """Set genome properties.
    #     this is used when breeding kids

    #     Args:
    #         genome (dict): The genome parameters
    #     IMPROVE
    #     """
    #     self.parents = [mom_ID, dad_ID]
    #     self.geneparam = geneparam
    #     self.update_hash()

    def train(self, train_and_score):
        """Train the genome and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.accuracy == 0.0: # don't bother retraining ones we already trained
            self.accuracy = train_and_score(self)

    def print_genome(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%%", self.accuracy * 100)
        logging.info("UniID: %d", self.u_id)
        logging.info("Mom and Dad: %d %d", self.parents[0], self.parents[1])
        logging.info("Gen: %d", self.generation)
        logging.info("Hash: %s", self.hash)

    def print_genome_ma(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d",
                     self.accuracy * 100, self.u_id, self.parents[0],
                     self.parents[1], self.generation)
        logging.info("Hash: %s", self.hash)

    def print_geneparam(self):
        """Print nb_neurons as single list."""

        gen = self.geneparam.copy()
        nb_dense_neurons = []
        for i in range(1, gen['nb_dense_layers'] + 1):
            nb_dense_neurons.append(gen[f'__dense_neurons_{i}'])
        # replace individual layer numbers with single list
        gen['__dense_neurons'] = nb_dense_neurons

        nb_cnn_neurons = []
        for i in range(1, gen['nb_cnn_layers'] + 1):
            nb_cnn_neurons.append(gen[f'__cnn_neurons_{i}'])
        # replace individual layer numbers with single list
        gen['__cnn_neurons'] = nb_cnn_neurons

        logging.info(gen)
