# Module that provides basic tools to examine the evolution of the algorithms and evaluate the results obtained.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
import numpy as np
import pandas as pd
from ..algorithm.interface.MultiObjective import MOA
from ..error.ProgramExceptions import UnsuitableClassifier


class MultiObjectiveEvaluator:
    """
    Class that allows evaluating the performance of the best solutions found by a multi-objective algorithm.

    Methods
    --------
        metrics(): Function that computes several metrics. Currently available accuracy, recall,
            precision, and f1-score for train and test data.

        training_confusion_matrix(): Method that computes the confusion matrix on training data.

        test_confusion_matrix(): Method that computes the confusion matrix using cross-validation.

        roc(): Method that computes the ROC curve.
    """

    def __init__(self, algorithm: MOA):
        """
            __init__(algorithm)
        """
        if not isinstance(algorithm, MOA):
            raise TypeError("Algorithm must be a multi-objective GA. Provided: %s" % type(algorithm))

        self.algorithm = algorithm

    def metrics_plot(self,  idx: int = None, cv: int = 5, reps: int = 5, n_jobs: int = -1):
        """
        Method that computes several metrics. Currently computed accuracy, recall, precision, and f1-score for
        train and test data. Metrics are computed by several stratified cross-validation.

        Parameters
        ------------
        :param idx: int
            Solution index.
        :param cv: int
            Number of K for stratified cross-validation. By default 5.
        :param reps: int
            Number of repetitions for k-cv. By default 5.
        :param n_jobs <optional> int
            Number of cores used for compute the repeated stratified cross-validation.
        """
        # Cross-validation
        from sklearn.model_selection import RepeatedStratifiedKFold
        # Scores
        from sklearn.metrics import make_scorer
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score

        # Define scores
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score),
                   'f1-score': make_scorer(f1_score)}

        # Define cross-validation
        k_cv = RepeatedStratifiedKFold(n_splits=cv, n_repeats=reps)
        x_data, y_data = self.algorithm.get_dataset()

        best_solutions = self._get_pareto_front()

        # If there is only one solution on the pareto front the representation changes
        if len(best_solutions) == 1: idx = 0

        # If no particular solution representation is selected, represent all
        if idx is None:
            self._evaluate_all_solutions_metrics(best_solutions, x_data, y_data, scoring, k_cv, n_jobs)
        # Represent the solution selected by the user
        else:
            if idx > len(best_solutions):
                raise Exception("Index out of bounds.")

            self._evaluate_solution_metrics(idx, best_solutions, x_data, y_data, scoring, k_cv, n_jobs)

    def metrics_table(self,  idx: int = None, cv: int = 5, reps: int = 5, n_jobs: int = -1):
        """
        Method that computes several metrics. Currently computed accuracy, recall, precision, and f1-score for
        train and test data. Metrics are computed by several stratified cross-validation.

        Parameters
        ------------
        :param idx: int
            Solution index.
        :param cv: int
            Number of K for stratified cross-validation. By default 5.
        :param reps: int
            Number of repetitions for k-cv. By default 5.
        :param n_jobs <optional> int
            Number of cores used for compute the repeated stratified cross-validation.
        """
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.model_selection import cross_validate
        from sklearn.metrics import make_scorer
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import f1_score

        # Define scores
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score),
                   'f1-score': make_scorer(f1_score)}

        # Define cross-validation
        k_cv = RepeatedStratifiedKFold(n_splits=cv, n_repeats=reps)
        x_data, y_data = self.algorithm.get_dataset()

        # Perform evaluation
        best_solutions = self._get_pareto_front()
        scores = cross_validate(
            estimator=self.algorithm.fitness[0].estimator, X=x_data[:, best_solutions[idx]], y=y_data,
            scoring=scoring, cv=k_cv, n_jobs=n_jobs, return_train_score=True)

        # Remove irrelevant data
        del scores['score_time']
        del scores['fit_time']

        # Define the tables
        score_mean = [np.mean(score) for score in scores.values()]
        score_mean_train = score_mean[1::2]
        score_mean_test = score_mean[::2]
        score_std = [np.std(score) for score in scores.values()]
        score_std_train = score_std[1::2]
        score_std_test = score_std[0::2]
        df_train = round(pd.DataFrame(score_mean_train+score_std_train,index=['accuracy_mean','precision_mean','recall_mean','f1_mean','accuracy_std','precision_std','recall_std','f1_std']),3).transpose()
        df_test = round(pd.DataFrame(score_mean_test+score_std_test,index=['accuracy_mean','precision_mean','recall_mean','f1_mean','accuracy_std','precision_std','recall_std','f1_std']),3).transpose()
        return(df_train,df_test)

    def predictions(self, idx: int = None, cv: int = 5, n_jobs: int = -1):
        """ 
        Function that performs predictions through cross validation

        Parameters
        ------------
        :param idx: int
            Solution index.
        :param cv: int
            Number of K for stratified cross-validation. By default 5.
        :param n_jobs <optional> int
            Number of cores used for compute the repeated stratified cross-validation.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_predict
        k_cv = StratifiedKFold(n_splits=cv,shuffle=True)
        x_data, y_data = self.algorithm.get_dataset()
        best_solutions = self._get_pareto_front()
        predictions = cross_val_predict(estimator=self.algorithm.fitness[0].estimator,X=x_data[:, best_solutions[idx]],y=y_data,cv=k_cv,n_jobs=n_jobs)
        return(predictions)

    def training_confusion_matrix(self, idx: int = None, fitness_idx: int = 0):
        """
        Function that computes the confusion matrix on the training data. By indicating an index (idx parameter)
        you can select one of the solutions from the non-dominated front.
        If the algorithm has more than one fitness function with estimators (in addition to optimization in the
        number of features), the estimator can be selected with the parameter fitness_idx.

        Parameters
        -----------
        :param idx: int
            Index of the solution to represent.
        :param fitness_idx: int
            Index of the fitness function to represent.
        """
        # Get Pareto front
        best_solutions = self._get_pareto_front()

        # Get dataset
        x_data, y_data = self.algorithm.get_dataset()

        #  If there is only one solution on the pareto front the representation changes
        if len(best_solutions) == 1:
            idx = 0

        #  If no particular solution representation is selected, represent all
        if idx is None:
            self._evaluate_all_solutions_cm_train(best_solutions, x_data, y_data, fitness_idx)

        # Represent the solution selected by the user
        else:
            if idx > len(best_solutions):
                raise Exception("Index out of bounds.")

            self._evaluate_solution_cm_train(idx, best_solutions, x_data, y_data, fitness_idx)

    def test_confusion_matrix(self, idx: int = None, cv: int = 5, reps: int = 5,
                              fitness_idx: int = 0, random_state: int = None):
        """
        Compute the confusion matrix using cross-validation repeats. By indicating an index (idx parameter)
        you can select one of the solutions from the non-dominated front.
        If the algorithm has more than one fitness function with estimators (in addition to optimization in the
        number of features), the estimator can be selected with the parameter fitness_idx.

        Parameters
        -----------
        :param idx: int
            Index of the solution to represent.
        :param cv: int
            Number of folds in which to split the data.
        :param reps: int
            Number of repetitions for CV.
        :param fitness_idx: int
            Index of the fitness function to represent.
        :param random_state: int
        """
        # Get Pareto front
        best_solutions = self._get_pareto_front()

        #  If there is only one solution on the pareto front the representation changes
        if len(best_solutions) == 1:
            idx = 0

        #  If no particular solution representation is selected, represent all
        if idx is None:
            self._evaluate_all_solutions_cm_test(best_solutions, cv, reps, fitness_idx, random_state)

        # Represent the solution selected by the user
        else:
            if idx > len(best_solutions):
                raise Exception("Index out of bounds.")

            self._evaluate_solution_cm_test(idx, best_solutions, cv, reps, fitness_idx, random_state)

    def roc(self, idx: int = None, estimator=None, fitness_idx: int = 0, cv: int = 5, reps: int = 1,
            positive_class: int = None, random_state: int = 0):
        """
        Function that allows to represent the ROC curve on the solutions found in the non-dominated front
        using cross validation with repetitions.

        Parameters
        ------------
        :param idx: int
            Index of the solution to represent.
        :param fitness_idx: int
            Index of the fitness function to represent.
        :param estimator: <optional> sklearn.base.BaseEstimator
            If none is provided the algorithm estimator will be used. This must support predictions with
            probabilities, otherwise it will throw an exception.
        :param cv: <optional> int
            Default 5
        :param reps: <optional> int
            Default 1
        :param positive_class: <optional> int
            By default the class selected as positive in the algorithm. In the case that the algorithm does
            not have a positive class and one is not provided, an exception will be thrown.
        :param random_state: <optional> int
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
        from sklearn.metrics import roc_curve, auc

        # Get positive class
        if positive_class is None:
            positive_class = self.algorithm.positive_class

        # Check if the estimator can be used to compute roc curves
        if estimator is None:
            estimator = self.algorithm.fitness[fitness_idx].estimator
            try:
                estimator.probability = True
            except:
                raise UnsuitableClassifier(
                    "The classifier does not support probabilities, therefore the ROC curve cannot be computed. "
                    "Run the algorithm with a classifier that supports probabilities or provide a valid classifier "
                    "that support probabilities using the argument \"estimator\".")

        # Get dataset
        x_data, y_data = self.algorithm.get_dataset()

        # Get non-dominated solutions
        best_solutions = self._get_pareto_front()

        # if the user has selected a certain solution use only that solution
        if idx is not None:
            indexes = [idx]
        else:
            indexes = [n for n in range(len(best_solutions))]

        fig, ax = plt.subplots(figsize=(10, 5))

        viridis = plt.cm.get_cmap('viridis', len(indexes))

        solutions_legend = []

        for index in indexes:

            mean_tp = 0.0
            mean_fp = np.linspace(0, 1, 100)
            roc_auc = []

            #  Create cross-validation iterator
            cv_iterator = list(
                RepeatedStratifiedKFold(
                    n_splits=cv, n_repeats=reps, random_state=random_state).split(
                    x_data[:, best_solutions[index]], y_data
                )
            )
            

            # Compute the ROC curve for each fold of each solution
            for i, (train, test) in enumerate(cv_iterator):

                probs = estimator.fit(x_data[np.ix_(train, best_solutions[index])], y_data[train])\
                                 .predict_proba(x_data[np.ix_(test, best_solutions[index])])

                fp, tp, thresholds = roc_curve(y_data[test], probs[:, 1], pos_label=positive_class)

                mean_tp += np.interp(mean_fp, fp, tp)
                mean_tp[0] = 0.0

                # Compute AUC
                roc_auc.append(auc(fp, tp))

                ax.plot(fp, tp, color=viridis(index), alpha=0.3)

            solutions_legend.append(mlines.Line2D(
                [], [], color=viridis(index), marker='.', markersize=5,
                label='Solution (%d) AUC = %.3f +/- %.3f' % (index, np.mean(roc_auc), np.std(roc_auc))))

        ax.plot([0, 1], [0, 1],
                linestyle='-.',
                color='black',
                label="Random Classifier")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')

        plt.legend(handles=solutions_legend, loc="lower right")

        plt.show()

    def _get_pareto_front(self):
        """
        Method that allows obtaining the fitness from the non-dominated front of solutions.

        Returns
        ---------
        :return: list
            Solutions fitness
        """
        # Get individuals in Pareto front
        best_individuals = [individual.features for individual in self.algorithm.population.individuals
                            if individual.fitness.rank == 0]

        return best_individuals

    def _process_title(self, idx):
        """
        Method that formats the title of the representations.

        Returns
        ---------
        :return: str
        """
        #  Format title
        features_length = len(self.algorithm.best_features[idx])
        if features_length > 5:

            title = ", ".join(self.algorithm.best_features[idx][:int(features_length/2)])
            title += "\n"
            title += ", ".join(self.algorithm.best_features[idx][int(features_length/2):])
        else:
            title = ", ".join(self.algorithm.best_features[idx])

        return title

    def _evaluate_all_solutions_metrics(self, best_solutions, x_data, y_data, scoring, k_cv, n_jobs):
        """
        Method that evaluates all solutions reached by a multi-objective algorithm using repeated cross-validation.
        """
        from sklearn.model_selection import cross_validate
        # Visualization
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(len(best_solutions), figsize=(12, 5 * len(best_solutions)))
        fig.subplots_adjust(wspace=0.5, hspace=1)

        for i, solution in enumerate(best_solutions):
            # Perform evaluation (using first fitness estimator)
            scores = cross_validate(
                estimator=self.algorithm.fitness[0].estimator, X=x_data[:, solution], y=y_data,
                scoring=scoring, cv=k_cv, n_jobs=n_jobs, return_train_score=True)

            # Remove irrelevant data
            del scores['score_time']
            del scores['fit_time']

            # Visualization
            viridis = plt.cm.get_cmap('viridis', len(scores))
            score_metrics = list(scores.keys())
            score_mean = [np.mean(score) for score in scores.values()]
            score_std = [np.std(score) for score in scores.values()]

            for n, (metric, score) in enumerate(zip(score_metrics, score_mean)):
                # This condition makes the color map the same in training and test and different
                # between measurements.
                if n % 2 == 0:
                    c_map_idx = n
                ax[i].bar(x=metric, yerr=score_std[n], height=score,
                          color=viridis(c_map_idx), alpha=0.7, label=metric)

            for n in range(len(score_mean)):
                ax[i].text(x=n - 0.4, y=(score_mean[n] + score_std[n] + 0.02),
                           s="%.3f +/- %.3f" % (score_mean[n], score_std[n]), size=8)

            ax[i].set_ylim(bottom=0.5, top=1.0)
            ax[i].legend(bbox_to_anchor=(1.15, 0.25), loc='lower right', ncol=1)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].set_title('(%d) %s' % (i, self._process_title(idx=i)))
            ax[i].title.set_position([.5, 1.2])

        plt.show()

    def _evaluate_solution_metrics(self, idx, best_solutions, x_data, y_data, scoring, k_cv, n_jobs):
        """
        Method that evaluates a solution indicated by the user using repeated cross-validation.
        """
        from sklearn.model_selection import cross_validate
        # Visualization
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1, figsize=(12, 5))

        # Perform evaluation (using first fitness estimator)
        scores = cross_validate(
            estimator=self.algorithm.fitness[0].estimator, X=x_data[:, best_solutions[idx]], y=y_data,
            scoring=scoring, cv=k_cv, n_jobs=n_jobs, return_train_score=True)

        # Remove irrelevant data
        del scores['score_time']
        del scores['fit_time']

        # Visualization
        viridis = plt.cm.get_cmap('viridis', len(scores))
        score_metrics = list(scores.keys())
        score_mean = [np.mean(score) for score in scores.values()]
        score_std = [np.std(score) for score in scores.values()]

        for n, (metric, score) in enumerate(zip(score_metrics, score_mean)):
            # This condition makes the color map the same in training and test and different
            # between measurements.
            if n % 2 == 0:
                c_map_idx = n
            ax.bar(x=metric, yerr=score_std[n], height=score, color=viridis(c_map_idx), alpha=0.7, label=metric)

        for n in range(len(score_mean)):
            ax.text(x=n - 0.4, y=(score_mean[n] + score_std[n] + 0.02),
                    s="%.3f +/- %.3f" % (score_mean[n], score_std[n]), size=8)

        ax.set_ylim(bottom=0.5, top=1.0)
        ax.legend(bbox_to_anchor=(1.15, 0.25), loc='lower right', ncol=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(self._process_title(idx))
        ax.title.set_position([.5, 1.2])

        plt.show()

    def _evaluate_all_solutions_cm_train(self, best_solutions, x_data, y_data, fitness_idx: int):
        """
        Compute the confusion matrix for each of the solutions in non-dominated front on training data.
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Define font
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 16}

        fig, ax = plt.subplots(len(best_solutions), figsize=(5, 5 * len(best_solutions)))
        fig.subplots_adjust(hspace=0.5)

        for i, solution in enumerate(best_solutions):

            # Calculate confusion matrix
            fitted_estimator = self.algorithm.fitness[fitness_idx].estimator.fit(X=x_data[:, solution], y=y_data)
            y_pred = fitted_estimator.predict(x_data[:, solution])
            confmat = confusion_matrix(y_true=y_data, y_pred=y_pred)

            # Plot confusion matrix
            ax[i].matshow(confmat, cmap=plt.cm.Blues, alpha=0.5)
            for ii in range(confmat.shape[0]):
                for jj in range(confmat.shape[1]):
                    ax[i].text(x=jj, y=ii, s=confmat[ii, jj],
                               va='center', ha='center',
                               fontdict=font)

            ax[i].set_title('(%d) %s' % (i, self._process_title(idx=i)))
            ax[i].set_xlabel('Predicted label')
            ax[i].set_ylabel('True label')
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])

        plt.show()

    def _evaluate_solution_cm_train(self, idx, best_solutions, x_data, y_data, fitness_idx: int):
        """
        Compute the confusion matrix of the selected solution in the non-dominated front on training data.
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Define font
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 16}

        fig, ax = plt.subplots(figsize=(5, 5))

        fitted_estimator = self.algorithm.fitness[fitness_idx].estimator.fit(X=x_data[:, best_solutions[idx]], y=y_data)
        y_pred = fitted_estimator.predict(x_data[:, best_solutions[idx]])
        confmat = confusion_matrix(y_true=y_data, y_pred=y_pred)

        # Plot confusion matrix
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.5)

        for ii in range(confmat.shape[0]):
            for jj in range(confmat.shape[1]):
                ax.text(x=jj, y=ii, s=confmat[ii, jj],
                        va='center', ha='center',
                        fontdict=font)

        ax.set_title(self._process_title(idx))
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    def _evaluate_all_solutions_cm_test(self, best_solutions, cv, reps, fitness_idx, random_state):
        """
        Compute the confusion matrix for each of the solutions in non-dominated front using repeated CV.
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Define font
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 16}

        fig, ax = plt.subplots(len(best_solutions), figsize=(5, 5 * len(best_solutions)))

        fig.subplots_adjust(hspace=0.5)

        for i, solution in enumerate(best_solutions):

            # Calculate confusion matrix
            confmat_mean, confmat_std = self._compute_matrix(solution=solution, cv=cv, reps=reps,
                                                             fitness_idx=fitness_idx, random_state=random_state)

            # Plot confusion matrix
            ax[i].matshow(confmat_mean, cmap=plt.cm.Blues, alpha=0.5)

            for ii in range(confmat_mean.shape[0]):
                for jj in range(confmat_mean.shape[1]):
                    ax[i].text(x=jj, y=ii, s="%.2f\n(+/- %.2f)" % (confmat_mean[ii, jj], confmat_std[ii, jj]),
                               va='center', ha='center', fontdict=font)

            ax[i].set_title('(%d) %s' % (i, self._process_title(idx=i)))
            ax[i].set_xlabel('Predicted label')
            ax[i].set_ylabel('True label')
            ax[i].set_yticklabels([])
            ax[i].set_xticklabels([])

        plt.show()

    def _evaluate_solution_cm_test(self, idx, best_solutions, cv, reps, fitness_idx: int, random_state: int):
        """
        Compute the confusion matrix of the selected solution in the non-dominated front using repeated CV.
        """
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt

        # Define font
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 16}

        fig, ax = plt.subplots(figsize=(5, 5))

        # Calculate confusion matrix
        confmat_mean, confmat_std = self._compute_matrix(solution=best_solutions[idx], cv=cv, reps=reps,
                                                         fitness_idx=fitness_idx, random_state=random_state)

        # Plot confusion matrix
        ax.matshow(confmat_mean, cmap=plt.cm.Blues, alpha=0.5)

        for ii in range(confmat_mean.shape[0]):
            for jj in range(confmat_mean.shape[1]):
                ax.text(x=jj, y=ii, s="%.2f\n(+/- %.2f)" % (confmat_mean[ii, jj], confmat_std[ii, jj]),
                        va='center', ha='center', fontdict=font)

        ax.set_title(self._process_title(idx))
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    def _compute_matrix(self, solution, cv, reps, fitness_idx, random_state):
        """
        Private function that performs the computation of several confusion matrices dividing the data in train and
        test according to the parameters specified by the user.

        Parameters
        ------------
        :param split_test float
            Percentage of data that for test set.
        :param reps int
            Number of repetitions. If split_test is not specified, this parameter will be ignored.
        :param random_states list
            Random seeds.
        """
        import numpy as np
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import RepeatedStratifiedKFold

        # Get dataset created from best features
        x_data, y_data = self.algorithm.get_dataset()
        x_data = x_data[:, solution]

        rep_k_fold = RepeatedStratifiedKFold(n_splits=cv, n_repeats=reps, random_state=random_state)

        confmats = []

        for train, test in rep_k_fold.split(x_data, y_data):

            # Fit using training data
            fitted_estimator = self.algorithm.fitness[fitness_idx].estimator.fit(X=x_data[train], y=y_data[train])

            # Predict using test data
            y_pred = fitted_estimator.predict(x_data[test])

            # Compute confusion matrix
            confmats.append(confusion_matrix(y_true=y_data[test], y_pred=y_pred))

        return np.array(np.mean(confmats, axis=0)), np.array(np.std(confmats, axis=0))
