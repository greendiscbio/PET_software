# Module that contains the implementation of the populations used in the algorithms
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## External dependencies
import numpy as np
from random import randrange
from copy import deepcopy
## Module dependencies
from .Individual import Individual
from ..error.ProgramExceptions import *


class Population:
    """
    Class that represent the basic population implementation.
    """
    def __init__(self, size: int, min_feat_init: int = 2, max_feat_init: int = None):
        """
            __init__(size: int, min_feat_init: int = 2, max_feat_init: int = None)
        """
        self.min_feat_init = min_feat_init
        self.max_feat_init = max_feat_init
        self.size = size
        self._individuals = None
        self._features = None

    def __repr__(self):
        return f"Population(size={self.size}, min_feat_init={self.min_feat_init}, max_feat_init={self.max_feat_init})"

    def __str__(self):
        return self.__repr__()

    @property
    def length(self):
        """
        Returns the current size of the population (note that it is not the same as the size).

        Returns
        --------
        :return int
        """
        if self._individuals is None:
            return 0

        return len(self._individuals)

    @property
    def individuals(self):
        """
        Returns the individuals (list of list of features)

        Returns
        ---------
        :return 2d-array
        """
        return self._individuals

    @property
    def fitness(self):
        """
        DESCRIPTION
        :return:
        """
        return [individual.fitness for individual in self._individuals]

    @property
    def features(self):
        """
        Return population features

        Returns
        --------
        :return list
        """
        return self._features

    @features.setter
    def features(self, new_features: list):
        """
        Select new population features.

        Parameters
        -----------
        :param new_features: list
        """
        if self.max_feat_init is None:
            self.max_feat_init = len(new_features)

        self._features = new_features

    @property
    def features_num(self):
        """
        Return features indices

        Returns
        --------
        :return list
        """
        return list(self._features.keys())

    def init(self, include_all: bool = True):
        """
        Initializes a random population.

        Parameters
        -----------
        :param include_all: bool
            Indicates if all features will be included in the population initialization.
        """
        self._individuals = self._population_init(self.size, include_all=include_all)

    def set_new_individuals(self, new_individuals: list):
        """
        Update population individuals. New individuals list must be:

            [(individual, fitness), (individual, fitness)]

        Parameters
        -----------
        :param new_individuals: list
        """
        self._individuals = new_individuals

    def init_features(self, features: list):
        """
        Function that assigns to the numbers with which the genes (features) in the algorithm are encoded
        the real name of the feature. If no features are provided, the default names will be the column number.

        Parameters
        -------------
        :param features: list
            Features in the same order as the columns of the predictor variables.
        """
        # Encode features in integers and save the names
        self._features = {n: col for n, col in enumerate(features)}

        if len(self._features) <= self.min_feat_init:
            warning = "Minimum number of features in the random initialization of the population" \
                      " incompatible with the total number of features."
            raise InconsistentParameters(warning)

        if self.max_feat_init is None:
            self.max_feat_init = len(self._features)

        if len(self._features) < self.max_feat_init:
            warning = "Maximum number of features in the random initialization of the population" \
                      " incompatible with the total number of features."
            raise InconsistentParameters(warning)

    def copy(self):
        """
        Return a deep copy of the current population.

        Returns
        --------
        :return Population
        """
        return deepcopy(self)

    def pop_individual(self, idx: int):
        """
        Take and remove an individual from population by index.

        Parameters
        -----------
        :param idx: int

        Returns
        --------
        :return tuple
            ([Features], fitness)
        """
        return self._individuals.pop(idx)

    def get_individual(self, idx: int):
        """
        Takes an individual from population by index.

        Parameters
        -----------
        :param idx: int

        Returns
        --------
        :return tuple
            ([Features], fitness)
        """
        return self._individuals[idx]

    def pop_random_individual(self):
        """
        Take and remove an individual from population randomly.

        Returns
        --------
        :return tuple
            ([Features], fitness)
        """
        idx = randrange(len(self._individuals))

        return self.pop_individual(idx)

    def get_random_individual(self):
        """
        Take an individual from population randomly without remove it.

        Returns
        --------
        :return tuple
            ([Features], fitness)
        """
        idx = randrange(len(self._individuals))

        return self.get_individual(idx)

    def generate_random_population(self, size: int):
        """
        This method generates a new random population with a given size.

        Parameters
        -----------
        :param size: int

        Returns
        --------
        :return Population
        """
        if size == 0:
            return None

        random_individuals = self._population_init(size=size, include_all=False)

        # Create new population
        random_population = Population(size=len(random_individuals))
        random_population.set_new_individuals(random_individuals)
        random_population.features = self.features

        return random_population

    def create_new_population(self, size: int, individuals: list):
        """
        Method that creates a new population using a given size and individuals.

        Parameters
        -----------
        :param size: int
        :param individuals tuple
            ([features], fitness)

        Returns
        --------
        :return Population
        """
        new_population = Population(size=size, max_feat_init=self.max_feat_init, min_feat_init=self.min_feat_init)
        new_population.set_new_individuals(individuals)
        new_population.features = self.features

        return new_population

    @staticmethod
    def merge_population(population_1, population_2):
        """
        Method that takes two populations and mixes them.

        Parameters
        -----------
        :param population_1: Population
        :param population_2: Population

        Returns
        ---------
        :return Population
        """
        # Merge individuals and their fitness
        individuals = population_1.individuals + population_2.individuals

        # Sum population sizes
        size = population_1.size + population_2.size

        # Merge population features
        features = {**population_1.features, **population_2.features}

        # Create new population
        mixed_population = Population(size=size)

        # Assign individuals and their fitness
        mixed_population.set_new_individuals(individuals)

        # Assign features
        mixed_population.features = features

        return mixed_population

    def _population_init(self, size: int, include_all: bool = True, num_feats: int = None):
        """
        Function that generates a random population based on the size received as argument. By default, all features
        are included in the population unless the include_all variable is specified as false.

        Parameters
        -------------
        :param size: int
            Population size to be generated.
        :param include_all: bool
            True to include all features.
        :param num_feats: int
            Number of features that the returned population will have.

        Returns
        ----------
        :return: tuple
            ([[features], [features], ...], [fitness, fitness, ...])
        """
        initial_population = []

        all_feats = np.array(self.features_num)

        # It's possible to balance the number of features eliminated from the population providing a num_feats
        if num_feats is not None:
            num_genes = int(num_feats / size)

        for n in range(size):
            if num_feats is None:
                # A minimum individual size can be selected by user
                num_genes = randrange(self.min_feat_init, self.max_feat_init)

            # Individual initialization
            individual_feat = np.random.choice(all_feats, size=num_genes, replace=False)
            # Append to population
            initial_population.append(individual_feat)

        # For first initialization
        if include_all:
            # Check the features that are not in the population and add them
            features_used = np.concatenate(initial_population, axis=0)
            not_used_features = np.setdiff1d(all_feats, features_used)
            for feature in not_used_features:
                r_num = randrange(size)
                initial_population[r_num] = np.append(initial_population[r_num], [feature])

        return [Individual(individual_feat) for individual_feat in initial_population]
