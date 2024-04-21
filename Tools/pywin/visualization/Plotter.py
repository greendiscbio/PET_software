# Module that provides basic tools to examine the evolution of the algorithms and evaluate the results obtained.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## External dependencies
import numpy as np
from math import inf as infinite
import warnings
## Module dependencies
from ..wrapper.interface.WrapperBase import WrapperBase
from ..wrapper.GlobalizationGA import GlobalizationGA
from ..algorithm.BasicGA import BasicGA
from ..algorithm.interface import MOA


class Plotter:
    """
    Class that allows to represent the genetic algorithm evolution and performance.

    Methods
    -----------
    plot_evolution(): Represents information of the number of features and performance of the model in
        each generation.
    """

    @classmethod
    def plot_evolution(cls, *algorithms):
        """
        Function that performs the graphic representation of the training process of the algorithm(s).
        Based on the type of algorithm received as a parameter it will make a different representation.
        """
        # Get a list of estimators
        mono_objective_algorithms, multi_objective_algorithms = [], []

        for algorithm in algorithms:
            #  To extract the algorithms in wrapper.
            if isinstance(algorithm, WrapperBase):

                if isinstance(algorithm, GlobalizationGA):
                    for sub_algorithm in algorithm.algorithms:
                        # Warning message for hypervolume calculation in  GlobalizationGA
                        if isinstance(sub_algorithm, MOA):
                            warnings.warn(
                                "The hypervolume is a relative measure, the GlobalizationGA algorithm reduces "
                                "the number of features at each batch, therefore this measure is not representative "
                                "of the evolution of the algorithm. You can pass each algorithm a function to evaluate "
                                "the number of features with a fixed value to avoid this behavior.")
                            break

                for wrapper_alg in algorithm.algorithms:
                    if isinstance(wrapper_alg, BasicGA):
                        mono_objective_algorithms.append(wrapper_alg)
                    if isinstance(wrapper_alg, MOA):
                        multi_objective_algorithms.append(wrapper_alg)
            else:
                if isinstance(algorithm, BasicGA):
                    mono_objective_algorithms.append(algorithm)
                if isinstance(algorithm, MOA):
                    multi_objective_algorithms.append(algorithm)

        # Call plotting functions
        if len(mono_objective_algorithms) != 0:
            Plotter._plot_mono_objective(mono_objective_algorithms)

        if len(multi_objective_algorithms) != 0:
            Plotter._plot_multi_objective(multi_objective_algorithms)

    @classmethod
    def _plot_mono_objective(cls, algorithms):
        """
        Graphic representation of:

            - The evolution of the total fitness in population.
            - The evolution of the best individual.
            - The number of features in population.
            - The number of features in best individual.
            - The performance measure evolution.
        """
        import matplotlib.pyplot as plt

        scores, generations, y_tot_fit, y_best_fit = [], [], [], []
        y_mean_feat, y_std_feat, y_best_feat = [], [], []
        y_best_performance, y_mean_performance, y_std_performance = [], [], []
        y_best_idx_max, x_best_idx_max, y_tot_idx_max, x_tot_idx_max = [], [], [], []

        # Get parameters
        for n, algorithm in enumerate(algorithms):
            args, score = algorithm.training_evolution()
            generations.append(list(args['total_fitness'].keys()))
            y_tot_fit.append(list(args['total_fitness'].values()))
            y_best_fit.append(list(args['best_fitness'].values()))
            y_mean_feat.append(list(args['mean_num_features'].values()))
            y_std_feat.append(list(args['std_num_features'].values()))
            y_best_feat.append(list(args['best_num_features'].values()))
            y_best_performance.append(list(args['best_%s' % score].values()))
            y_mean_performance.append(list(args['mean_%s' % score].values()))
            y_std_performance.append(list(args['std_%s' % score].values()))
            scores.append(score)

        # Get maximum values
        for n in range(len(generations)):
            y_best_idx_max.append(y_best_fit[n].index(max(y_best_fit[n])))
            x_best_idx_max.append(generations[n][y_best_idx_max[n]])
            y_tot_idx_max.append(y_tot_fit[n].index(max(y_tot_fit[n])))
            x_tot_idx_max.append(generations[n][y_tot_idx_max[n]])

        # Plot estimator(s) evolution
        fig, ax = plt.subplots(6, figsize=(12, 40))
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        cmap = plt.cm.get_cmap('viridis', len(generations))

        for n, algorithm in enumerate(algorithms):
            # Subplot 0 -> Population evolution
            ax[0].plot(generations[n], y_tot_fit[n], color=cmap(n),
                       label="Population fitness (%s)" % algorithm.id )
            ax[0].annotate('Maximum: %.3f' % max(y_tot_fit[n]), xy=(x_tot_idx_max[n], max(y_tot_fit[n])),
                           xytext=(x_tot_idx_max[n], max(y_tot_fit[n]) * 1.0025),
                           arrowprops=dict(facecolor='black', shrink=0.05))

            # Subplot 1 -> Best fitness in population per generation
            ax[1].plot(generations[n], y_best_fit[n], color=cmap(n),
                       label="Best fitness (%s)" % algorithm.id)
            ax[1].annotate('Maximum: %.3f' % max(y_best_fit[n]), xy=(x_best_idx_max[n], max(y_best_fit[n])),
                           xytext=(x_best_idx_max[n], max(y_best_fit[n]) * 1.025),
                           arrowprops=dict(facecolor='black', shrink=0.05))

            # Subplot 2 -> Number of features evolution in population
            ax[2].plot(generations[n], y_mean_feat[n], color=cmap(n),
                       label="Num. feats in population (%s)" % algorithm.id)
            ax[2].fill_between(generations[n],
                               np.array(y_mean_feat[n]) - (np.array(y_std_feat[n])),
                               np.array(y_mean_feat[n]) + (np.array(y_std_feat[n])),
                               alpha=0.1, color=cmap(n))

            # Subplot 3 -> Number of features evolution in best individual per generation
            ax[3].plot(generations[n], y_best_feat[n], color=cmap(n),
                       label="Num. feats best indv. (%s)" % algorithm.id)

            # Subplot 4 -> Best score in population per generation
            ax[4].plot(generations[n], y_best_performance[n],
                       label="Best %s (%s)" % (scores[n], algorithm.id), color=cmap(n))

            # Subplot 5 -> Score evolution in population
            ax[5].plot(generations[n], y_mean_performance[n],
                       label="Average %s (%s)" % (scores[n], algorithm.id), color=cmap(n))
            ax[5].fill_between(generations[n],
                               np.array(y_mean_performance[n]) - (np.array(y_std_performance[n])),
                               np.array(y_mean_performance[n]) + (np.array(y_std_performance[n])),
                               alpha=0.1, color=cmap(n))

        # Get limits
        y_tot_fit_min, y_tot_fit_max, y_best_fit_min, y_best_fit_max = infinite, 0, infinite, 0
        for tot_fit, best_fit in zip(y_tot_fit, y_best_fit):
            if min(tot_fit) < y_tot_fit_min:
                y_tot_fit_min = min(tot_fit)
            if max(tot_fit) > y_tot_fit_max:
                y_tot_fit_max = max(tot_fit)
            if min(best_fit) < y_best_fit_min:
                y_best_fit_min = min(best_fit)
            if max(best_fit) > y_best_fit_max:
                y_best_fit_max = max(best_fit)

        # Subplot 0 -> Population evolution
        ax[0].set_ylim(bottom=y_tot_fit_min, top=y_tot_fit_max * 1.01)
        ax[0].set_title("Total fitness in population")
        ax[0].legend(loc="lower right")

        # Subplot 1 -> Best fitness in population per generation
        ax[1].set_ylim(bottom=y_best_fit_min, top=y_best_fit_max * 1.2)
        ax[1].set_title("Best fitness in population / generation")
        ax[1].legend()

        # Subplot 2 -> Number of features evolution in population
        ax[2].set_title("Mean number of features in population")
        ax[2].legend()

        # Subplot 3 -> Number of features evolution in best individual per generation
        ax[3].set_title("Number of features in best individual")
        ax[3].legend()

        # Subplot 4 -> Best score in population per generation
        ax[4].set_title("Best %s per generation" % scores[0])
        ax[4].set_ylim(bottom=0.5, top=1.0)
        ax[4].legend()

        # Subplot 5 -> Score evolution in population
        ax[5].set_title("Average %s in population" % scores[0])
        ax[5].set_ylim(bottom=0.5, top=1.0)
        ax[5].legend()

        # Remove borders and add grid to all graphics
        for subplot in ax:
            subplot.grid(True, alpha=0.3)
            subplot.spines['top'].set_visible(False)
            subplot.spines['right'].set_visible(False)

        plt.show()

    @classmethod
    def _plot_multi_objective(cls, algorithms):
        """
        Graphic representation of:

            - Hypervolume indicator.
            - Number of solutions on the non-dominated Pareto front.
            - Best values achieved in each objective function.
        """
        import matplotlib.pyplot as plt

        generations, hypervolumes, num_solutions, best_values = [], [], [], []

        for algorithm in algorithms:
            algorithm_stats, scores = algorithm.training_evolution()
            generations.append(list(algorithm_stats['hypervolume'].keys()))
            hypervolumes.append(list(algorithm_stats['hypervolume'].values()))
            num_solutions.append(list(algorithm_stats['num_solutions_front'].values()))
            best_values.append(list(algorithm_stats['best_values'].values()))

        # Plot estimator(s) evolution
        fig, ax = plt.subplots(3, figsize=(12, 20))
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        linestyles = ['-', ':', '-.', '--']
        cmap = plt.cm.get_cmap('viridis', len(algorithms))

        for n, algorithm in enumerate(algorithms):
            # Subplot 0 -> Hypervolume
            ax[0].plot(generations[n], hypervolumes[n], color=cmap(n),
                       label="Algorithm: (%s)" % algorithm.id)

            # Subplot 1 -> Number of solutions in Pareto front per generation
            ax[1].plot(generations[n], num_solutions[n], color=cmap(n),
                       label="Algorithm: (%s)" % algorithm.id)

            # Subplot 2 -> Best values for each objective
            for i, score in enumerate(scores):
                ax[2].plot(generations[n], [value[i] for value in best_values[n]], color=cmap(n),
                           linestyle=linestyles[i], label="Score: (%s) (%s)" % (score, algorithm.id))

        # Get limits
        min_hyper, max_hyper, max_num_sol = 1, 0, 0

        for volumes, num_sols in zip(hypervolumes, num_solutions):
            if min(volumes) < min_hyper: min_hyper = min(volumes)
            if max(volumes) > max_hyper: max_hyper = max(volumes)
            if max(num_sols) > max_num_sol: max_num_sol = max(num_sols)

        # Subplot 0 ->  Hypervolume
        ax[0].set_ylim(bottom=min_hyper, top=max_hyper+0.001)
        ax[0].set_title("Hypervolume")
        ax[0].legend(loc="lower right")

        # Subplot 1 -> Best fitness in population per generation
        ax[1].set_ylim(bottom=0, top=max_num_sol + 1)
        ax[1].set_title("Num. solutions in Pareto front")
        ax[1].legend()

        # Subplot 2 -> Best value for each objective
        ax[2].set_ylim(bottom=0.5, top=1)
        ax[2].set_title("Best scores per generation")
        ax[2].legend()

        for subplot in ax:
            subplot.grid(True, alpha=0.3)
            subplot.spines['top'].set_visible(False)
            subplot.spines['right'].set_visible(False)

        plt.show()
