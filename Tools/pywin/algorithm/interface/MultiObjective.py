# Module that defines the interface that will be common to all the algorithms of the module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## Dependencies
import numpy as np
import random
from abc import ABCMeta, abstractmethod
## Module dependencies
from .._algorithm_init import _MOA_init
from .GenAlgBase import GenAlgBase
from pywin.population.Population import Population
from pywin.error.ProgramExceptions import *


def features_function(individual, total_feats: int = None):
    """
    Default function to optimize the number of features: 1 - (num ind. Feat. / num total feat.)
    """
    return 1 - (len(individual) / total_feats)


class MOA(GenAlgBase):
    """
     Base class for all multi-objective algorithms
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        """
            __init__(**kwargs)
        """
        # Call superclass
        super().__init__(**kwargs)

        # Check parameters consistency
        kwargs = _MOA_init(kwargs)

        self.fitness = kwargs['fitness']
        self.selection = kwargs['selection']
        self.mutation = kwargs.get('mutation', None)
        self.mutation_rate = kwargs.get('mutation_rate', None)
        self.crossover = kwargs['crossover']
        self.optimize_features = kwargs.get('optimize_features', False)
        self.features_function = kwargs.get('features_function', features_function)

        # Monitoring parameters
        self._evolution = {
            "hypervolume": {},
            "num_solutions_front": {},
            "best_values": {}
        }

    @property
    def best_features(self):
        """
        Get features from non dominated Pareto front.

        Returns
        ---------
        :return: list
            List with the selected features in each of the solutions of the non-dominated front.
        """
        # Get individuals in Pareto front
        pareto_front = [self._population.individuals[idx] for idx in range(self._population.length)
                        if self._population.fitness[idx].rank == 0]

        # Get feature names
        selected_features = []
        for individual in pareto_front:
            selected_features.append([self._population.features[idx] for idx in individual])

        return selected_features

    @property
    def best_performance(self):
        """
        Returns the best values achieved for each of the objective functions.

        Returns
        --------
        :return: list
            List with the best function values in each of the solutions of the non-dominated front.
        """
        scores_ = [solution.values for solution in self._population.fitness if solution.rank == 0]

        #  Separate scores into lists
        scores_ = list(zip(*scores_))

        # Add score name
        scores = {f"{self.fitness[n].score}({n})": scores_[n] for n in range(len(self.fitness))}

        # If the features have been optimized add them
        if self.optimize_features:
            scores['feature_scores'] = scores_[-1]
            scores['num_features'] = [len(features) for features in self.best_features]

        return scores

    def predict(self, X: np.ndarray):
        """
        Method NOT available for multi-objective algorithms
        """
        print("Method not available for MultiObjective algorithms.")
        pass

    @abstractmethod
    def get_dataset(self):
        """
        Function that returns a dataset with the selected features and associated class labels.
        """
        pass

    @abstractmethod
    def training_evolution(self):
        """
        Method that returns the data collected throughout the algorithm search.

        Returns
        ---------
        :return dict
        """
        return dict()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Method that starts the execution of the algorithm.

            fit(X: np.ndarray, y: np.ndarray)

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables.
        :param y: 1d-array
            Class labels.
        """
        pass

    @abstractmethod
    def continue_training(self, generations: int):
        """
        This method allows to continue training the algorithm since the last generation, over the indicated
        number of generations.

            continue_training(generations: int)

        Parameters
        -----------
        :param generations: int
        """
        pass
