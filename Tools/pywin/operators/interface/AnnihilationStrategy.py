# Module that defines the interface that will be common to all operators of the module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
from abc import ABCMeta, abstractmethod
from pywin.population.Population import Population


class AnnihilationStrategy:
    """
    Base class for annihilation strategies.
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def annihilate(population: Population, annihilation: float):
        """
        Method that takes a Population and eliminates the worst individuals using their fitness value.

        Parameters
        ------------
        :param population: Population
        :param annihilation: float
        """
        pass
