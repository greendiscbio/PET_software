# Module that defines wrappers that allow executing several algorithms in parallel.
# 
# Author: Fernando García <ga.gu.fernando@gmail.com> 
#
## External dependencies
import random
import numpy as np
import warnings
import multiprocessing as mp
# Module dependencies
from .Parallel import Parallel
from ..algorithm.interface import MOA


class GlobalizationGA(Parallel):
    """
    Algorithm that allows executing several algorithms in parallel, bringing together the best solutions every
    certain number of generations (called batches) and reinitializing the populations using all the combination of
    the features present in the best solutions.
    """

    def __init__(self, *algorithms, num_batches: int = 1, verbose: bool = False):
        """
            __init__(*algorithms, num_batches: int = 1, verbose: bool = False)

        Note
        -----
        The number of batches determines the number of times the best solutions achieved by the algorithms will
        intersect.
        """
        super().__init__(*algorithms)

        # Number of intersections between populations
        self.num_batches = num_batches

        # To select the generation batches
        self.batch_size = int(self._algorithms[0].generations / num_batches)

        for algorithm in self._algorithms.values():
            algorithm.generations = self.batch_size

        self.verbose = verbose

    def __repr__(self):
        return f"GlobalizationGA({list(self._algorithms.values())})"

    def __str__(self):
        return self.__repr__()

    def fit(self, X, y):
        """
        Function that calls the fit() method of each of the algorithms provided as parameters. Consult the
        documentation of each of the algorithms for more information.

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables
        :param y: 1d-array
            Class labels

        Returns
        ----------
        :return: Parallel
            Models fitted.
        """
        # Fit first batch
        super().fit(X, y)

        for batch in range(self.num_batches - 1):
            if self.__select_features():
                super().continue_training(self.batch_size)
            else:
                print("ALGORITHM CONVERGENCE: The number of features has been reduced to 2. "
                      "Impossible to continue training.")
                return self

        return self

    @property
    def best_performance(self):
        """
        Returns a list with the performance achieved by each algorithm

        Returns
        ---------
        :return: list
        """
        return [algorithm.best_performance for algorithm in self.algorithm]

    def continue_training(self, generations):
        """
        Function that calls the continue_training() method of each of the previously trained algorithms.
        """
        warnings.warn("WARNING: The algorithm will continue training in parallel without any intersection.")
        super().continue_training(generations)

    def __select_features(self):
        """
        Private method that obtains the best solutions reached by each algorithm and selects the total of different
        features as new features of the individual algorithms by re-initializing their populations.
        """
        # Get the features of the best solutions found by each of the algorithms.

        selected_feats = list()

        for algorithm in self._algorithms.values():

            for features in algorithm.best_features:
                #  If the algorithm is multi-objective, it is necessary to reduce the
                # number of features by one dimension
                if isinstance(algorithm, MOA):
                    for feature in features:
                        selected_feats.append(feature)
                else:
                    selected_feats.append(features)

        # Remove repeated features
        selected_feats = list(set(selected_feats))

        #  If the number of features is equal to or less than two, the algorithm ends
        if len(selected_feats) <= 2:
            return False

        # Create new features subset
        pop_feat_keys = list(self._algorithms[0].population.features.keys())
        pop_feat_values = list(self._algorithms[0].population.features.values())

        new_features = {
            pop_feat_values.index(feature): feature for feature in selected_feats if feature in pop_feat_values}

        # Console output
        if self.verbose:
            print("Number of different features: %d" % len(new_features))
            print("Features: %r" % list(new_features.values()))

        # Update features
        for algorithm in self._algorithms.values():
            algorithm.population.features = new_features
            algorithm._restart_population()

        return True
