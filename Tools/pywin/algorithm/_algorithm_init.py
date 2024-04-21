# Module that contains all the genetic algorithms implemented in the pywin module.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
import warnings
from scipy.spatial.distance import euclidean, minkowski, cosine, canberra
# Module dependencies
from ..operators.interface.ElitismStrategy import ElitismStrategy
from ..operators.interface.AnnihilationStrategy import AnnihilationStrategy
from ..operators.interface.MutationStrategy import MutationStrategy
from ..operators.interface.SelectionStrategy import SelectionStrategy
from ..operators.interface.CrossOverStrategy import CrossOverStrategy
from ..fitness.interface.FitnessStrategy import FitnessStrategy
from ..population.Population import Population
from ..imputation.interface.ImputationStrategy import ImputationStrategy
from ..error.ProgramExceptions import *
# Default
from ..operators.AnnihilateWorse import AnnihilateWorse
from ..operators.RandomMutation import RandomMutation
from ..operators.TournamentSelection import TournamentSelection
from ..operators.OnePoint import OnePoint
from ..operators.BestFitness import BestFitness
from ..fitness.MonoObjectiveCV import MonoObjectiveCV


def _GenAlgBase_init(kwargs: dict):
    """
    Function that verifies that all the necessary arguments for the initialization of the GenAlgBase
    algorithm have been provided.
    """
    if (kwargs.get('population', None) is not None) and (not isinstance(kwargs.get('population'), Population)):
        raise InconsistentParameters(
            "Incorrect population parameter, please provide a valid population (using Population), or, "
            "instead, define parameters: population_size and optionally max_feat_init/min_feat_init.")

    if not isinstance(kwargs.get('population'), Population):
        if kwargs.get('population_size', None) is None:
            raise InconsistentParameters(
                "Parameter population_size must be defined as an integer greater than 10 or, "
                "instead, provide a population using Population instance.")

        elif kwargs['population_size'] < 10:
            raise InconsistentParameters(
                "Parameter population_size must be defined as an integer greater than 10 or, "
                "instead, provide a population using Population instance.")

        else:
            # Create a population
            kwargs['population'] = Population(size=kwargs['population_size'],
                                              min_feat_init=kwargs.get('min_feat_init', 2),
                                              max_feat_init=kwargs.get('max_feat_init', None))

    # Check if the imputation strategy is valid.
    if kwargs.get('imputer', None) is not None:
        if not isinstance(kwargs['imputer'], ImputationStrategy):
            raise InconsistentParameters("Invalid imputation strategy type. Use a valid ImputationStrategy class.")

    # Check number of generations
    if kwargs.get('generations', 0) <= 0:
        raise InconsistentParameters("Incorrect generations parameter, it must be an positive integer greater than 0.")

    # Check random state
    if not isinstance(kwargs.get('random_state', 1), int) or kwargs.get('random_state', 1) < 1:
        raise InconsistentParameters("random_state must be an integer greater than 1.")

    return kwargs


def _BasicGA_init(kwargs: dict):
    """
    Function that verifies that all the necessary arguments for the initialization of the BasicGA
    algorithm have been provided.
    """
    # Check elitism parameter
    if kwargs.get('elitism', None) is not None:
        if isinstance(kwargs['elitism'], float) and (0 < kwargs['elitism'] < 1):
            kwargs['elitism_rate'] = kwargs['elitism']
            kwargs['elitism'] = BestFitness
        else:
            if not isinstance(kwargs['elitism'], ElitismStrategy):
                raise InconsistentParameters(
                    "Incorrect elitism parameter, please provide a valid elitism strategy (ElitismStrategy). Or "
                    "indicates a percentage (float). You can also enter a percentage using the elitism_rate parameter")

            elif kwargs.get('elitism_rate', None) is None:
                warnings.warn("Elitism strategy provided but no elitism rate. Select a valid elitism_rate")

            elif kwargs['elitism_rate'] <= 0 or kwargs['elitism_rate'] >= 1:
                raise InconsistentParameters(
                    "Percentage of elitism out of bounds. Select an elitism rate between 0 and 1.")

    # Check annihilation parameter
    if kwargs.get('annihilation', None) is not None:

        if isinstance(kwargs['annihilation'], AnnihilationStrategy):
            if kwargs.get('annihilation_rate', None) is not None:
                if kwargs['annihilation_rate'] < 0 or kwargs['annihilation_rate'] > 1:
                    raise InconsistentParameters("The annihilation parameter must be within the range 0 - 1")
            else:
                warnings.warn(
                    "Annihilation strategy provided but no annihilation rate. Select a valid annihilation_rate")

        else:
            if isinstance(kwargs['annihilation'], float):
                if 1 > kwargs['annihilation'] > 0:
                    kwargs['annihilation_rate'] = kwargs['annihilation']
                    kwargs['annihilation'] = AnnihilateWorse
                else:
                    raise InconsistentParameters(
                        "Provides a valid annihilation strategy (for example AnnihilateWorse() from pywin.operators) "
                        "or select a suitable value for the annihilation or annihilation_rate parameter within the "
                        "range 0 - 1")
            else:
                raise InconsistentParameters(
                    "Provides a valid annihilation strategy (for example AnnihilateWorse() from pywin.operators) "
                    "or select a suitable value for the annihilation or annihilation_rate parameter within the "
                    "range 0 - 1")

        if kwargs.get('fill_with_elite', None) is None:
            kwargs['fill_with_elite'] = 0

        elif not 0 <= kwargs['fill_with_elite'] <= 1:
            raise InconsistentParameters("The parameter fill_with_elite must be within the range 0 - 1")

    # Check mutation parameter
    if kwargs.get('mutation_rate', None) is not None:

        if isinstance(kwargs['mutation_rate'], int):
            raise InconsistentParameters("Parameter mutation_rate must be a number between 0 and 1")

        if isinstance(kwargs['mutation_rate'], float):

            if not (1 > kwargs['mutation_rate'] > 0):
                raise InconsistentParameters("Parameter mutation_rate must be a number between 0 and 1")
            else:
                kwargs['mutation'] = RandomMutation
        else:
            if isinstance(kwargs.get('mutation', None), float):
                kwargs['mutation_rate'] = kwargs['mutation']
                kwargs['mutation'] = RandomMutation

    # Check selection parameter
    if kwargs.get('selection', None) is None:
        # Default selection strategy TournamentSelection
        kwargs['selection'] = TournamentSelection(k=2, replacement=False, winners=1)
    else:
        if not isinstance(kwargs['selection'], SelectionStrategy):
            raise InconsistentParameters("Invalid selection technique. Use a valid SelectionStrategy class.")

    # Check cross-over parameter
    if kwargs.get('crossover', None) is None:
        # Default crossover strategy OnePoint
        kwargs['crossover'] = OnePoint()
    else:
        if not isinstance(kwargs['crossover'], CrossOverStrategy):
            raise InconsistentParameters("Invalid cross-over strategy type. Use a valid CrossOverStrategy class.")

    # Check fitness parameter
    if not isinstance(kwargs.get('fitness', None), FitnessStrategy):
        if kwargs.get('cv', None) is None:
            raise InconsistentParameters("You must provide a valid cross-validatior iterator "
                                         "in cv parameter (sklearn.BaseCrossValidator).")

        if kwargs.get('fitness', None) is None:
            raise InconsistentParameters("You must provide a valid fitness function or a FitnessStrategy in fitness "
                                         "parameter (sklearn.base.BaseEstimator or pywin.operators.FitnessStrategy).")

        # Create MonoObjectiveCV using accuracy as default metric
        kwargs['fitness'] = MonoObjectiveCV(
            estimator=kwargs['fitness'], cv=kwargs['cv'], score=kwargs.get('score', 'accuracy'),
            n_jobs=kwargs.get('n_jobs', 1))

    return kwargs


def _MOA_init(kwargs: dict):
    """
    DESCRIPTION
    :param kwargs:
    :return:
    """
    if kwargs.get('fitness', None) is None:
        raise InconsistentParameters(
            "You must provide a fitness value (for example MonoObjectiveCV from pywin.fitness).")
    else:
        if not isinstance(kwargs['fitness'], list):
            # Only one metric to optimize
            if not kwargs.get('optimize_features', False):
                raise InconsistentParameters("Only one metric has been provided for optimization.")
            # Wrong optimize_features parameter
            elif not isinstance(kwargs['optimize_features'], bool):
                raise InconsistentParameters("Parameter optimize_features must be True or False.")
            # Add fitness function to list
            else:
                kwargs['fitness'] = [kwargs['fitness']]
        else:
            if len(kwargs['fitness']) == 1 and not kwargs.get('optimize_features', False):
                raise InconsistentParameters("Only one metric has been provided for optimization.")

            for fitness_func in kwargs['fitness']:
                if not isinstance(fitness_func, FitnessStrategy):
                    raise InconsistentParameters(
                        f"Invalid fitness strategy, required FitnessStrategy provided: {type(fitness_func)}")

    # Check selection parameter
    if kwargs.get('selection', None) is None:
        # Default selection strategy TournamentSelection
        kwargs['selection'] = TournamentSelection(k=2, replacement=False, winners=1)
    else:
        if not isinstance(kwargs['selection'], SelectionStrategy):
            raise InconsistentParameters("Invalid selection technique. Use a valid SelectionStrategy class.")

    # Check mutation parameter
    if kwargs.get('mutation_rate', None) is not None:
        if isinstance(kwargs['mutation_rate'], float):
            if kwargs['mutation_rate'] >= 1 or kwargs['mutation_rate'] < 0:
                raise InconsistentParameters("mutation_rate must be a number between 0 and 1")
            else:
                kwargs['mutation'] = RandomMutation
        else:
            if not isinstance(kwargs.get('mutation', None), MutationStrategy):
                kwargs['mutation'] = RandomMutation

    # Check cross-over parameter
    if kwargs.get('crossover', None) is None:
        # Default selection strategy TournamentSelection
        kwargs['crossover'] = OnePoint()
    else:
        if not isinstance(kwargs['crossover'], CrossOverStrategy):
            raise InconsistentParameters("Invalid cross-over strategy type. Use a valid CrossOverStrategy class.")

    # Check the user-defined function for optimizing the number of features.
    if kwargs.get('features_function', None) is not None and callable(kwargs['features_function']):
        individual = [1, 2, 3]
        try:
            # Check if the function receives a variable called "individual"
            return_value = kwargs['features_function'](individual, 1)
        except:
            raise FeaturesFunctionError(
                "Impossible to evaluate the number of features of a solution with the provided function. Provide a "
                "valid function for parameter features_function or leave it as default. The function must receive an"
                " \"individual\" parameter and return a single numerical value to be maximized")

        # Check if the return value is a single value
        if not (isinstance(return_value, int) or isinstance(return_value, float)):
            raise FeaturesFunctionError(
                "Impossible to evaluate the number of features of a solution with the provided function. Provide a "
                "valid function for parameter features_function or leave it as default. The function must receive an"
                " \"individual\" parameter and return a single numerical value to be maximized")

    return kwargs


def _SPEA2_init(kwargs: dict):
    """
    Function that verifies that all the necessary arguments for the initialization of the NSGA2
    algorithm have been provided.
    """
    AVAILABLE_DISTANCES = ['euclidean', 'minkowski', 'cosine', 'canberra']

    if kwargs.get('distance', None) is not None:
        if isinstance(kwargs['distance'], str):
            input_distance = kwargs['distance'].lower()

            if input_distance == 'euclidean':
                kwargs['distance'] = euclidean
            elif input_distance == 'minkowski':
                kwargs['distance'] = minkowski
            elif input_distance == 'cosine':
                kwargs['distance'] = cosine
            elif input_distance == 'canberra':
                kwargs['distance'] = canberra
            else:
                raise InconsistentParameters("Distance to estimate the density of solutions, invalid, "
                                             "available distances: %s" % ", ".join(AVAILABLE_DISTANCES))
        else:
            raise InconsistentParameters("Distance to estimate the density of solutions, invalid, "
                                         "available distances: %s" % ", ".join(AVAILABLE_DISTANCES))
    else:
        kwargs['distance'] = euclidean

    if kwargs.get('archive_length', None) is not None:
        if kwargs['archive_length'] < 5:
            raise InconsistentParameters("The length of the file cannot be less than 5")

    return kwargs
