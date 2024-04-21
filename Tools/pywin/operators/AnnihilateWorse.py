# Module that defines the operators used by the algorithms to modify the populations of solutions.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## External dependencies
import numpy as np
## Module dependencies
from ..population.Population import Population
from .interface.AnnihilationStrategy import AnnihilationStrategy


class AnnihilateWorse(AnnihilationStrategy):
    """
    Eliminate individuals who have a lower fitness value.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "AnnihilateWorse"

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def annihilate(population: Population, annihilation: float):
        """
        This function eliminates the worst individuals according to the selected "annihilation" parameter.

        Parameters
        ------------
        :param population: Population
        :param annihilation: float
        """
        annihilation_size = int(population.length * annihilation)

        individuals = population.individuals

        # Sort the individuals
        sorted_by_fitness = sorted(individuals, key=lambda individual: individual.fitness)

        # Eliminate the worst individuals
        new_population = sorted_by_fitness[annihilation_size:]

        # Create new population
        surviving_population = population.create_new_population(
            size=population.size - annihilation_size, individuals=new_population
        )

        return surviving_population
