# Module containing the fitness functions used to evaluate individuals.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## External dependencies
import numpy as np
from functools import reduce
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.base import BaseEstimator
## Module dependencies
from .interface.FitnessStrategy import FitnessStrategy


class Hypervolume(FitnessStrategy):
    """
    Class that calculates an individual's fitness based on various metrics using a volume metric and reference
    point.
    To calculate this volume this class uses cross validation metrics from scikit-learn cross_validation(). The
    metrics  must be compatible with the estimators.
    """
    def __init__(self, estimator: BaseEstimator, cv: BaseCrossValidator,
                 scores: dict, features: bool = False, n_jobs: int = 1):
        """
        __init__(self, estimator: BaseEstimator, cv: BaseCrossValidator, scores: dict,
            features: bool = False, n_jobs: int = 1)

        Notes
        -------
        The scores variable must be a dictionary where the keys correspond to the measurements and the values to
        the reference point. If features is true the number of features will be considered to calculate the volume.
        """
        Hypervolume._check_init(estimator=estimator, cv=cv, scores=scores)

        self.features = features
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self._scores = list(scores.keys())
        self._references = list(scores.values())
        self._scores_repr = 'Hypervolume of %s' % ", ".join(scores)

        if self.features:
            self._scores_repr += ', NUM. FEATURES'
            self._references.append(1)

    def __repr__(self):
        return f"HyperVolume(estimator={str(self.estimator)} cv={str(self.cv)} scores={str(self._scores)} "\
               f"n_jobs={str(self.n_jobs)})"

    def __str__(self):
        return self.__repr__()

    @property
    def score(self):
        """
        Return scores representation.

        Returns
        ---------
        :return str
        """
        return self._scores_repr

    def eval_fitness(self, X: np.ndarray, y: np.ndarray, num_feats: int):
        """
        Function that evaluates the quality of a solution using the estimator and the cross-validation strategy
        provided and calculating the hypervolume covered by the solution. If the parameter features has been
        selected as true, the number of features of the individual is also taken into account.

        Parameters
        ------------
        :param X: 2d-array
            Predictor variables
        :param y: 1d-array
            Class labels
        :param num_feats: int
            Total number of features

        Returns
        ---------
        :return float
        """
        scores = cross_validate(estimator=self.estimator, X=X, y=y, return_train_score=False,
                                cv=self.cv, n_jobs=self.n_jobs, scoring=self._scores)

        # Get scores of interest
        scores = [scores['test_%s' % score].tolist() for score in self._scores]

        # Normalize num of features
        if self.features:
            num_feats = 1 - (X.shape[1] / num_feats)
            scores.append([num_feats for n in range(len(scores[0]))])

        return self._volume(scores)

    def fit(self, X, y):
        """
        Fit the estimator.

        Parameters
        -----------
        :param X: 2d-array
            Predictor variables.
        :param y: 1d-array
            Class labels

        Returns
        ---------
        :return sklearn.base.BaseEstimator
        """
        fitted_estimator = self.estimator.fit(X, y)

        return fitted_estimator

    def _volume(self, scores):
        """
        Function that calculates the hypervolume of the solution with respect to the anti-optimal. If a score
        is higher than the reference value, the reference value will be used instead of the value itself to
        calculate the volume.

        Parameters
        -----------
        :param scores: list

        Returns
        --------
        :return list
        """
        volumes = []
        for measures in zip(*scores):
            side = []
            # Get distances respect to anti-optimal point
            for n in range(len(self._references)):
                # Calculate the sides from the hypercube respect to the anti-optimal
                if measures[n] > self._references[n]: 
                    side.append(self._references[n])
                else: 
                    side.append(measures[n])

            # Calculate hypervolume
            volume = reduce(lambda a, b: a * b, side)

            volumes.append(volume)

        return np.mean(volumes)

    @staticmethod
    def _check_init(**kwargs):
        """
        Check that the parameters provided for the initialization are correct.

        Parameters
        ------------
        :param kwargs: dict
        """
        if not isinstance(kwargs.get('scores', None), dict):
            raise TypeError("Variable scores must be a dictionary. Provided: %s" % type(kwargs.get('scores', 'None')))

        if not isinstance(kwargs.get('estimator', None), BaseEstimator):
            raise TypeError("Variable estimator must be a sklearn.BaseEstimator. "
                            "Provided: %s" % type(kwargs.get('estimator', 'None')))

        if not isinstance(kwargs.get('cv', None), BaseCrossValidator):
            raise TypeError(
                "Variable cv must be a sklearn.BaseCrossValidator. Provided: %s" % type(kwargs.get('cv', 'None')))
