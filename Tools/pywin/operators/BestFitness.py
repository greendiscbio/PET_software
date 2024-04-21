# Module that defines the operators used by the algorithms to modify the populations of solutions.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## External dependencies
import numpy as np
## Module dependencies
from ..population.Population import Population
from .interface.ElitismStrategy import ElitismStrategy


class BestFitness(ElitismStrategy):
    """
    Create a new population using the individuals with the highest fitness value.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "BestFitness"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def select_elite(population: Population, elitism: float):
        """ FIX DESCRIPTION
        Function that selects the best individuals based on the elitism parameter.

        Parameters
        ------------
        :param population: Population
        :param elitism: float
            Percentage of individuals in the population that will form the elite.
        Returns
        ----------
        :return: Population
            Elite population
        """
        # Determine the elite size taking into account the elitism and population size.
        elite_length = int(population.length * elitism)

        individuals = population.individuals

        # Sort list and use the indices to get the best individuals (descending)
        sorted_by_fitness = sorted(individuals, key=lambda individual: individual.fitness, reverse=True)

        # Get elite
        elite_individuals = sorted_by_fitness[:elite_length]

        # Create elite population
        elite = population.create_new_population(size=elite_length, individuals=elite_individuals)

        return elite
