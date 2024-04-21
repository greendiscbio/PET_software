# Module that defines the interface that will be common to all operators of the module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
from abc import ABCMeta, abstractmethod
from pywin.population.Population import Population


class ElitismStrategy:
    """
    Base class for elitism strategies.
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def select_elite(population: Population, elitism: float):
        """
        Method that takes a population and select the best individuals (elite)

        Parameters
        ------------
        :param population: Population
        :param elitism: float
        """
        pass
