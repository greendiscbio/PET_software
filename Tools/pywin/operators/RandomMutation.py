# Module that defines the operators used by the algorithms to modify the populations of solutions.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## External dependencies
import numpy as np
## Module dependencies
from ..population.Population import Population
from .interface.MutationStrategy import MutationStrategy


class RandomMutation(MutationStrategy):
    """
    Introduce random mutations.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "RandomMutation"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def mutate(population: Population, mutation_rate: float):
        """
        Function that introduces mutations inside the population based on the established mutation rate using an
        uniform random distribution.
        """
        for individual in population.individuals:

            # Generates an array of the same length as the individual with probabilities between 0 and 1.
            prob = np.random.uniform(low=0.0, high=1.0, size=len(individual))
            positions = np.where(prob <= mutation_rate)[0]

            # If there aren't positions to mutate go to the next individual
            if len(positions) == 0:
                continue

            # Get not used features
            not_used_features = np.setdiff1d(np.array(population.features_num), individual.features)

            # If the individual has all features, mutations cannot be introduced
            if len(not_used_features) == 0:
                continue

            # If there are more positions to mutate than features, mutate with replacement
            if len(not_used_features) < len(positions):
                replace = True
            else:
                replace = False

            # Mutate individual
            individual.features[positions] = np.random.choice(not_used_features, size=len(positions), replace=replace)

            # If mutations have occurred with replacement, eliminate repeated features
            if replace:
                individual.features = np.unique(individual.features)

            # Shuffle individual to avoid bias
            np.random.shuffle(individual.features)

        return population
