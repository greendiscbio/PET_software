##########################################################################################################################################

import os
import re
import numpy
import pandas
import random
from itertools import product
from collections import Counter
from ast import literal_eval
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier                      
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import BayesianRidge
from pywin.algorithm import BasicGA, NSGA2, SPEA2
from pywin.population import Population, BlockPopulation
from pywin.fitness import MonoObjectiveCV, Hypervolume
from pywin.operators import TournamentSelection,RouletteWheel
from pywin.wrapper import Parallel, GlobalizationGA
from pywin.visualization import Plotter, Evaluator, MultiObjectiveEvaluator

##########################################################################################################################################

class Binarization:
    """
    Class that allows converting quantitative to binary data.
    #Methods
        get_binary(): converts a quantitative dataset into a binary dataset given a threshold.
    """
    
    def get_binary(self,Y,binary_threshold):
        """
        Function that converts a quantitative dataset into a binary dataset given a threshold.
        :param Y: (Dataframe) dataframe with quantitative data.
        :param binary_threshold: (Integer) threshold above which a value is considered to be positive.
        """
        Y_binary = (Y>binary_threshold).astype('int')
        if numpy.sum(Y_binary.values)==0: raise Exception('The dataframe has no possitive labels')        
        return(Y_binary)

##########################################################################################################################################

class Imputation:
    """
    Class that allows performing different imputation methods.
    #Methods
        imputate(): performs an imputation on a dataset given an imputation strategy.
    """   
    
    def imputate(self,X,imputation_strategy):
        """
        Function that performs an imputation on a dataset given an imputation strategy.
        :param X: (Dataframe) dataframe with some missing values.
        :param imputation_strategy: (String) technique to fill the missing values.
        """
        if imputation_strategy=='mean':
            X_imputed = X.fillna(X.mean())
        elif imputation_strategy=='median':
            X_imputed = X.fillna(X.median())
        elif imputation_strategy=='knn':
            knn_imputer = KNNImputer(n_neighbors=10, weights="uniform")
            X_imputed = pandas.DataFrame(knn_imputer.fit_transform(X))
            X_imputed.columns = X.columns
        elif imputation_strategy=='bayes':
            bayes_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=100)
            X_imputed = pandas.DataFrame(bayes_imputer.fit_transform(X))
            X_imputed.columns = X.columns
        return(X_imputed)

##########################################################################################################################################

class Transformations:
    """
    Class that includes different operations on the input data.
    #Methods
        encode_columns(): transforms a qualitative column into its dummy variables.
    """   

    def encode_column(self,X,column,transformations):
        """
        Function that transforms a qualitative column into its dummy variables.
        :param X: (Dataframe) dataframe with some qualitative variables.
        :param column: (String) column where the transformation will take place
        :param transformation: (Dictionary) dictionary with the values and its correspondant category.
        """
        X_transformed = X.replace({column:transformations})
        X_encoded = pandas.get_dummies(X_transformed)
        return(X_encoded)

##########################################################################################################################################

class Normalization:
    """
    Class that allows performing different normalization methods.
    #Methods
        normalise(): performs a normalisation on a dataset given a normalisation strategy.
    """   
    
    def normalise(self,X,normalisation_strategy):
        """
        Function that performs a normalisation on a dataset given a normalisation strategy.
        :param X: (Dataframe) dataframe with values to be normalised.
        :param normalisation_strategy: (String) technique to normalise the values.
        """
        if normalisation_strategy=='standard': scaler = StandardScaler()
        elif normalisation_strategy=='minmax': scaler = MinMaxScaler()
        X_normalised = pandas.DataFrame(scaler.fit_transform(X),columns=X.columns,index=X.index)
        return(X_normalised)

##########################################################################################################################################

class Models:
    """
    Class that includes different operations on the output data.
    #Methods
        keep_best(): keeps the best model for each brain region, given an optimization criteria.
        evaluate_metrics(): evaluates the metrics of models.
        evaluate_features(): evaluates the features of models.
        map_features(): creates a map with the features of the models.
    """   
  
    def keep_best(self,train,test,features,regions,criteria,file):
        """
        Function that keeps the best model for each brain region, given an optimization criteria.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model. 
        :param features: (Dataframe) dataframe with the features information for each model.
        :param regions: (List) list with all the regions to explore.
        :param criteria: (List) list with the metric(s) to optimize (if more than one metric is considered, they are sumed up) .
        :param file: (String) name that is given to the file where the results are stored.
        """
        df = test.copy()
        best_models = {}
        criteria_mean = [i + '_mean' for i in criteria]
        criteria_std = [i + '_std' for i in criteria]
        for region in regions:
            try:
                temp_df = df.loc[df['Model'].str.contains('_'+str(region)+'$')]
                temp_df['mean'] = temp_df[criteria_mean].sum(axis=1)
                temp_df['std'] = temp_df[criteria_std].sum(axis=1)
                temp_df['score'] = temp_df['mean']-temp_df['std']
                best_model = temp_df.iloc[temp_df['score'].argmax()]
                best_model_name = best_model['Model']
                best_model_score = best_model['score']
                best_models[best_model_name] = best_model_score
            except: pass
        best_models = sorted(best_models,key=best_models.get,reverse=True)
        best_train = train.loc[train['Model'].isin(best_models)]
        best_train['Model'] = pandas.Categorical(best_train['Model'],categories=best_models,ordered=True)
        best_train = best_train.sort_values('Model')
        best_train.to_csv(str(file)+'_best_train.csv',mode='a',index=False)
        best_test = test.loc[test['Model'].isin(best_models)]
        best_test['Model'] = pandas.Categorical(best_test['Model'],categories=best_models,ordered=True)
        best_test = best_test.sort_values('Model')
        best_test.to_csv(str(file)+'_best_test.csv',mode='a',index=False)
        best_features = features.loc[features['Model'].isin(best_models)]
        best_features['Model'] = pandas.Categorical(best_features['Model'],categories=best_models,ordered=True)
        best_features = best_features.sort_values('Model')
        best_features.to_csv(str(file)+'_best_features.csv',mode='a',index=False)

    def evaluate_metrics(self,train,test,region):
        """
        Function that evaluates the metrics of models.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model.
        :param region: (String) region, if specified, that will be filtered.
        """
        train = train.loc[train['Model'].str.contains('_'+str(region)+'$')]
        test = test.loc[test['Model'].str.contains('_'+str(region)+'$')]
        train_metrics_mean = train[['accuracy_mean','f1_mean','precision_mean','recall_mean']].transpose()
        train_metrics_mean.index = ['accuracy_train','f1_train','precision_train','recall_train']
        test_metrics_mean = test[['accuracy_mean','f1_mean','precision_mean','recall_mean']].transpose()
        test_metrics_mean.index = ['accuracy_test','f1_test','precision_test','recall_test']
        metrics_mean = pandas.concat([train_metrics_mean,test_metrics_mean],axis=0).sort_index(axis=0)  
        train_metrics_std = train[['accuracy_std','f1_std','precision_std','recall_std']].transpose()
        train_metrics_std.index = ['accuracy_train','f1_train','precision_train','recall_train']
        test_metrics_std = test[['accuracy_std','f1_std','precision_std','recall_std']].transpose()
        test_metrics_std.index = ['accuracy_test','f1_test','precision_test','recall_test']
        metrics_std = pandas.concat([train_metrics_std,test_metrics_std],axis=0).sort_index(axis=0)  
        stats_metrics = pandas.concat([metrics_mean,metrics_std],axis=1)
        stats_metrics.columns = ['Mean','Std']
        float_to_str = lambda flt: str(flt).ljust(5,"0")
        df = round(stats_metrics['Mean'],3).map(float_to_str) + '+/-' + round(stats_metrics['Std'],3).map(float_to_str)
        df = df.to_frame()
        df.columns = ['Region='+str(region)]
        return(df)

    def evaluate_features(self,features,features_all,region):
        """
        Function that evaluates the features of models.
        :param features: (Dataframe) dataframe with the features from the best models.
        :param features_all: (Dataframe) dataframe with the features from the best models.
        :param region: (String) region, if specified, that will be filtered.
        """
        features_best = literal_eval(list(features[features['Model'].str.contains('_'+str(region)+'$')]['Features'])[0])
        features_list = list(features_all[features_all['Model'].str.contains('_'+str(region)+'$')]['Features'])
        features_number = 0
        features_dictionary = {}
        for i in features_list:
            features_number+=len(literal_eval(i))
            for j in literal_eval(i):
                try: features_dictionary[j]+=1
                except: features_dictionary[j]=1
        features_number= round(features_number/len(features_list),2)
        for i in features_dictionary: features_dictionary[i]= round(features_dictionary[i]/len(features_list),2)
        features_ordered_list = Counter(features_dictionary).most_common()
        df = pandas.DataFrame({'Features best model':[features_best],'Features all models':[features_ordered_list],'Avg#':[features_number]})
        df.index = ['Region='+str(region)]
        return(df)
    
    def map_features(self,features_all,ordered_features,region):
        """
        Function that creates a map with the features of the models.
        :param features_all: (Dataframe) dataframe with the features from the best models.
        :param ordered_features: (List) list with the features in a defined order.
        :param region: (String) region, if specified, that will be filtered.
        """
        features_list = list(features_all[features_all['Model'].str.contains('_'+str(region)+'$')]['Features'])
        if features_list==[]: raise Exception('There is no model for the chosen region') 
        features_map = pandas.DataFrame(0,index=[region],columns=ordered_features)
        for i in features_list: 
            for j in literal_eval(i):
                features_map[j]+=1
        features_map = round(features_map/len(features_list),2)
        df = features_map
        return(df)    

##########################################################################################################################################