# Module that defines the interface that will be common to all operators of the module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
from abc import ABCMeta, abstractmethod
from pywin.population.Population import Population


class MutationStrategy:
    """
    Base class for mutation strategies.
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def mutate(population: Population, mutation_rate: float):
        """
        Method that receives a population and mutation rate and introduce variations in the individuals.

        Parameters
        ------------
        :param population: Population
        :param mutation_rate: float
        """
        pass
