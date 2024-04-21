# Module that provides basic tools to examine the evolution of the algorithms and evaluate the results obtained.
#
# Author: Fernando García <ga.gu.fernando@gmail.com>
#
## Module dependencies
import numpy as np
import pandas as pd
from ..algorithm.BasicGA import BasicGA
from ..error.ProgramExceptions import UnsuitableClassifier


class Evaluator:
    """
    Class that allows evaluating the performance of the best combination of features found by the genetic algorithm.

    Methods
    ---------
    plot_learning_curve(): Function that shows the performance of the algorithm using training and validation
        data based on the size of the training data.

    metrics(): Function that computes several metrics. Currently available accuracy, recall, precision,
        and f1-score for train and test data.

    training_confusion_matrix(): Method that computes the confusion matrix on training data.

    test_confusion_matrix(): Method that computes the confusion matrix using cross-validation.

    roc(): Method that computes the ROC curve.
    """

    def __init__(self, algorithm: BasicGA):
        """
            __init__(self, algorithm: BasicGA)
        """
        if isinstance(algorithm, BasicGA):
            self.algorithm = algorithm
        else:
            raise TypeError("The algorithm must belong to BasicGA. Provided: %s" % type(algorithm))

    def __repr__(self):
        return f"Evaluator(algorithm={self.algorithm})"

    def __str__(self):
        return self.__repr__()

    def plot_learning_curve(self, cv: int = 5, n_jobs: int = -1):
        """
        Function that shows the performance of the algorithm using training and validation data based on the size of
        the training data. This allows to analyze the variance and the bias of the algorithm.
        This method apply cross validation with stratification.

        Parameters
        -------------
        :param cv: int
            Number of k-folds. It must be a number appropriate to the number of instances in each of the classes.
        :param n_jobs: int
            Number of threads to compute cross validation.
        """
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve

        # Get dataset selected by the algorithm
        x_data, y_data = self.algorithm.get_dataset()

        # Compute learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=self.algorithm.fitness.estimator, X=x_data, y=y_data, train_sizes=np.linspace(0.05, 1.0, 20),
            cv=cv, n_jobs=n_jobs
        )

        # Compute values
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label="training accuracy")
        ax.fill_between(
            train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.3, color='blue'
        )

        ax.plot(
            train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label="Validation accuracy"
        )

        ax.fill_between(
            train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.1, color='green'
        )

        ax.set_xlabel("Training samples")
        ax.set_ylabel("Accuracy")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
        plt.show()

    def training_confusion_matrix(self):
        """
        Function that computes the confusion matrix on the training data.
        """
        from sklearn.metrics import confusion_matrix

        # Get dataset and true labels
        x_data, y_data = self.algorithm.get_dataset()

        #  Use training data for predictions
        y_pred = self.algorithm.predict(x_data)
        confmat = confusion_matrix(y_true=y_data, y_pred=y_pred)

        # Confusion matrix representation
        self._represent_cm(confmat)

    def test_confusion_matrix(self, cv: int = 5, reps: int = 5, random_state: int = None):
        """ 
        Function that computes the confusion matrix making repetitions of cross-validation (the results given are
        the average of each fold)

        Parameters
        ------------
        :param cv <optional> int
            Default 5.
        :param reps <optional> int
            Number of repetitions. Default 5.
        :param random_state <optional> int
            Random seed.
        """
        from sklearn.metrics import confusion_matrix

        # Compute average
        confmat_mean, confmat_std = self._compute_matrix(cv, reps, random_state)

        # Confusion matrix representation
        self._represent_cm(confmat_mean, confmat_std)

    def metrics_plot(self, cv: int = 5, reps: int = 5, n_jobs: int = -1):
        """ 
        Function that computes several metrics. Currently computed accuracy, recall, precision, and f1-score for
        train and test data. Metrics are computed by several stratified cross-validation.

        Parameters
        ------------
        :param cv: int
            Number of K for stratified cross-validation. By default 5.
        :param reps: int
            Number of repetitions for k-cv. By default 5.
        :param n_jobs <optional> int
            Number of cores used for compute the repeated stratified cross-validation.
        """
        # Cross-validation
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.model_selection import cross_validate
        # Scores
        from sklearn.metrics import make_scorer
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        from sklearn.metrics import auc  # This score raise an error (currently removed from scores)
        from sklearn.metrics import f1_score
        # Visualization
        import matplotlib.pyplot as plt

        # Define scores
        scoring = {'accuracy': make_scorer(accuracy_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score),
                   'f1-score': make_scorer(f1_score)}

        # Define cross-validation 
        k_cv = RepeatedStratifiedKFold(n_splits=cv, n_repeats=reps)
        x_data, y_data = self.algorithm.get_dataset()

        # Perform evaluation
        scores = cross_validate(
            estimator=self.algorithm.fitness.estimator, X=x_data, y=y_data,
            scoring=scoring, cv=k_cv, n_jobs=n_jobs, return_train_score=True)

        # Remove irrelevant data
        del scores['score_time']
        del scores['fit_time']

        # Visualization
        viridis = plt.cm.get_cmap('viridis', len(scores))
        score_metrics = list(scores.keys())
        score_mean = [np.mean(score) for score in scores.values()]
        score_std = [np.std(score) for score in scores.values()]

        fig, ax = plt.subplots(figsize=(12, 6))

        for n, (metric, score) in enumerate(zip(score_metrics, score_mean)):
            # This condition makes the color map the same in training and test and different between measurements.
            if n % 2 == 0:
                idx = n
            ax.bar(x=metric, yerr=score_std[n], height=score, color=viridis(idx), alpha=0.7, label=metric)

        for n in range(len(score_mean)):
            plt.text(x=n - 0.4, y=(score_mean[n] + score_std[n] + 0.02),
                     s="%.3f +/- %.3f" % (score_mean[n], score_std[n]), size=8)

        plt.xticks(rotation=25)
        plt.ylim(bottom=0.5, top=1.0)
        plt.legend(bbox_to_anchor=(1.15, 0.25), loc='lower right', ncol=1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.title('%s performance' % self.algorithm.id, fontsize='x-large')
        ttl = ax.title
        ttl.set_position([.5, 1.2])
        plt.show()

    def metrics_table(self, cv: int = 5, reps: int = 5, n_jobs: int = -1):
        """ 
        Function that computes several metrics. Currently computed accuracy, recall, precision, and f1-score for
        train and test data. Metrics are computed by several stratified cross-validation.

        Parameters
        ------------
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
        scores = cross_validate(
            estimator=self.algorithm.fitness.estimator, X=x_data, y=y_data,
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

    def predictions(self, cv: int = 5, n_jobs: int = -1):
        """ 
        Function that performs predictions through cross validation

        Parameters
        ------------
        :param cv: int
            Number of K for stratified cross-validation. By default 5.
        :param n_jobs <optional> int
            Number of cores used for compute the repeated stratified cross-validation.
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_predict
        k_cv = StratifiedKFold(n_splits=cv,shuffle=True)
        x_data, y_data = self.algorithm.get_dataset()
        predictions = cross_val_predict(estimator=self.algorithm.fitness.estimator,X=x_data,y=y_data,cv=k_cv,n_jobs=n_jobs)
        return(predictions)
    
    def roc(self, estimator=None, cv: int = 5, reps: int = 1, positive_class: int = None, random_state: int = 0):
        """
        Function that allows to represent the ROC curve on the best solution found using cross validation
        with repetitions.

        Parameters
        ------------
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
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
        from sklearn.metrics import roc_curve, auc

        #  Get positive class
        if positive_class is None:
            positive_class = self.algorithm.positive_class

        #  Check if the estimator can be used to compute roc curves
        if estimator is None:
            estimator = self.algorithm.fitness.estimator
            try:
                estimator.probability = True
            except:
                raise UnsuitableClassifier(
                    "The classifier does not support probabilities, therefore the ROC curve cannot be computed. "
                    "Run the algorithm with a classifier that supports probabilities or provide a valid classifier "
                    "that support probabilities using the argument \"estimator\".")

        # Get dataset
        x_data, y_data = self.algorithm.get_dataset()

        fig, ax = plt.subplots(figsize=(10, 5))

        viridis = plt.cm.get_cmap('viridis', 1)

        mean_tp = 0.0
        mean_fp = np.linspace(0, 1, 100)
        roc_auc = []

        #  Create cross-validation iterator
        cv_iterator = list(
            RepeatedStratifiedKFold(n_splits=cv, n_repeats=reps, random_state=random_state).split(x_data, y_data)
        )

        #  Compute the ROC curve for each fold of each solution
        for i, (train, test) in enumerate(cv_iterator):
            probs = estimator.fit(x_data[train], y_data[train]).predict_proba(x_data[test])

            fp, tp, thresholds = roc_curve(y_data[test], probs[:, 1], pos_label=positive_class)

            mean_tp += np.interp(mean_fp, fp, tp)
            mean_tp[0] = 0.0

            # Commpute AUC
            roc_auc.append(auc(fp, tp))

            ax.plot(fp, tp, color=viridis(0), alpha=0.3)

        legend = [mlines.Line2D(
            [], [], color=viridis(0), marker='.', markersize=5,
            label='AUC = %.2f +/- %.2f' % (np.mean(roc_auc), np.std(roc_auc)))]

        ax.plot([0, 1], [0, 1],
                linestyle='-.',
                color='black',
                label="Random Classifier")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic')

        plt.legend(handles=legend, loc="lower right")

        plt.show()

    @staticmethod
    def _represent_cm(confmat_mean: np.ndarray, confmat_std: np.ndarray = None):
        """
        Method that represents the confusion matrix.

        Parameters
        -----------
        :param confmat: 2d-array
            Confusion matrix (FP, FN, TN, TP)
        """
        import matplotlib.pyplot as plt

        # Define font
        font = {'family': 'serif',
                'weight': 'normal',
                'size': 16}

        # Confusion matrix visualization
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(confmat_mean, cmap=plt.cm.Blues, alpha=0.5)
        for i in range(confmat_mean.shape[0]):
            for j in range(confmat_mean.shape[1]):
                if confmat_std is None:
                    ax.text(x=j, y=i, s=confmat_mean[i, j],
                            va='center', ha='center',
                            fontdict=font)
                else:
                    ax.text(x=j, y=i, s="%.2f\n(+/- %.2f)" % (confmat_mean[i, j], confmat_std[i, j]),
                            va='center', ha='center', fontdict=font)

        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

    def _compute_matrix(self, cv: int, reps: int, random_state: int = None):
        """
        Private function that performs the computation of several confusion matrices dividing the data in train and
        test according to the parameters specified by the user.

        Parameters
        ------------
        :param cv int
        :param reps int
            Number of repetitions. If split_test is not specified, this parameter will be ignored.
        :param random_state 
        """
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import RepeatedStratifiedKFold

        # Get dataset created from best features
        x_data, y_data = self.algorithm._X, self.algorithm._y
        x_data = x_data[:, self.algorithm._evolution['best_features'][0]]

        rep_k_fold = RepeatedStratifiedKFold(n_splits=cv, n_repeats=reps, random_state=random_state)

        confmats = []

        for train, test in rep_k_fold.split(x_data, y_data):

            # Fit using training data
            self.algorithm._best_model.fit(x_data[train], y_data[train])

            # Predict using test data
            y_pred = self.algorithm._best_model.predict(x_data[test])

            # Compute confusion matrix
            confmats.append(confusion_matrix(y_true=y_data[test], y_pred=y_pred))

        return np.array(np.mean(confmats, axis=0)), np.array(np.std(confmats, axis=0))
