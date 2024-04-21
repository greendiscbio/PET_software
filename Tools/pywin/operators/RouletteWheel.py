# "pywin"
#
# Module that defines the operators used by the algorithms to modify the populations of solutions.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## External dependencies
import numpy as np
from functools import reduce
from random import uniform
## Module dependencies
from ..population.Population import Population
from .interface.SelectionStrategy import SelectionStrategy


class RouletteWheel(SelectionStrategy):
    """
    Class that implements the roulette wheel selection strategy applicable to genetic algorithms.
    Calculate the cumulative probability of survival of each individual based on their fitness.
    Individuals with a higher probability will have more opportunities to be randomly selected.
    If the fitness is not of the float type (for example in multi-objective algorithms) consider as
    fitness the sum of the values of the objective functions. (For the stability of the algorithm
    these values should be scaled between 0 and 1 for all target functions)
    """

    def __init__(self):
        pass

    @staticmethod
    def select(population: Population, new_pop_length: int):
        """
        Select individuals from the population. The best individuals are more likely to survive.

        Parameters
        ------------
        :param population Population
        :param new_pop_length int
            Size of the individuals that will form the new population

        Returns
        ----------
        :return: Population
        """
        # Get individuals fitness
        fitness = [individual.fitness for individual in population.individuals]

        # Get individuals probability
        survival_prob = list(map(RouletteWheel._prob_solution, fitness))

        # Get total probability
        total_prob = reduce(lambda a, b: a + b, survival_prob)

        # Calculate individual probability (accumulative)
        survival_prob = list(map(lambda a: a / total_prob, survival_prob))
        survival_prob = np.cumsum(survival_prob)

        selected_individuals = []

        while len(selected_individuals) < new_pop_length:
            # Get position
            individual_idx = np.where(survival_prob >= uniform(0, 1))[0][0]
            # Append to new individuals
            selected_individuals.append(population.get_individual(individual_idx))

        # Create the new population
        new_population = population.create_new_population(
            size=len(selected_individuals), individuals=selected_individuals
        )

        return new_population

    @staticmethod
    def _prob_solution(fitness):
        """
        Function that adds the values of various objective functions when the algorithm is multi-objective.
        If the fitness is not a numerical value it must be a Solution or "something" that implement the attribute
        values.
        """
        if not isinstance(fitness, float):
            return reduce(lambda a, b: a + b, fitness.values)

        return fitness
