# "pywin"
#
# Module that defines the operators used by the algorithms to modify the populations of solutions.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## External dependencies
import numpy as np
## Module dependencies
from ..population.Population import Population
from .interface.SelectionStrategy import SelectionStrategy


class TournamentSelection(SelectionStrategy):
    """
    Class that implements the tournament selection strategy applicable to genetic algorithms.
    """
    def __init__(self, k: int, replacement: bool = False, winners: int = 1):
        """
        __init__(k: int, replacement: bool = False, winners: int = 1)

        Parameters
        ------------
        :param k: int
            Number of the individuals selected for tournament.
        :param replacement: bool
            It indicates if the selection of individuals is going to be done with replacement (an individual can
            be selected multiple times) or without replacement.
        :param winners: int
            Number of individuals that are selected as tournament winners.
        """
        self.k = k
        self.replacement = replacement
        self.winners = winners

    def __repr__(self):
        return f"TournamentSelection(k={self.k} replacement={self.replacement} winners={self.winners})"

    def __str__(self):
        return self.__repr__()

    def select(self, population: Population, new_pop_length: int):
        """
        Function that applies the tournament selection algorithm. It takes a total of k individuals randomly
        (without replacement / with replacement) from the population and returns the best of those k individuals.

        Parameters
        ------------
        :param population Population
        :param new_pop_length int
            Size of the individuals that will form the new population

        Returns
        ----------
        :return: Population
            Tournament winners
        """
        winners = []
        for n in range(int(new_pop_length/self.winners)):
            # Get individual indices with or without replacement
            gladiators_idx = np.random.choice(
                range(population.length), size=self.k, replace=self.replacement).tolist()

            # Get individuals from population
            gladiators = [population.get_individual(idx) for idx in gladiators_idx]

            # Get fitness
            fitness = [individual.fitness for individual in gladiators]

            # Get winners
            for best in range(self.winners):
                idx = fitness.index(max(fitness))
                winners.append(gladiators.pop(idx))
                fitness.pop(idx)

        # Create a population using the winners
        best_individuals = population.create_new_population(size=len(winners), individuals=winners)

        return best_individuals
