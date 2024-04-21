# Module that defines the interface that will be common to all operators of the module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
from abc import ABCMeta, abstractmethod
from pywin.population.Population import Population


class CrossOverStrategy:
    """
    Base class for cross-over strategies.
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def cross_population(population: Population):
        """
        Method that receives a population and carry out the cross-over between the individuals.

        Parameters
        ------------
        :param population: Population
        """
        pass
