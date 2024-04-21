# Module that defines the interface that will be common to all operators of the module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
from pywin.population.Population import Population
from abc import ABCMeta, abstractmethod


class SelectionStrategy:
    """
    Base class for selection strategies.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def select(self, population: Population, new_pop_length: int):
        """
        This method receives a population, and the size of the new population that will be generated

        Parameters
        ------------
        :param population: Population
        :param new_pop_length: int

        Returns
        ----------
        :return: Population
        """
        pass
