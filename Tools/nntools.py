##########################################################################################################################################

import re
import numpy
import pandas
import random
import matplotlib
from math import sqrt
from os.path import isfile
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2,f_classif
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,KFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import BinaryAccuracy,Precision,Recall
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input,Dropout
from tensorflow.keras.initializers import Constant
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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
        return Y_binary
    
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
        return X_imputed
 
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
        
class FeatureSelection:
    """
    Class that allows performing different feature selection algorithms.
    #Methods
        pca_feature_ranking(): performs a feature ranking based on PCA.
        chi_feature_ranking(): performs a feature ranking based on chi scores.
        fvalue_feature_ranking(): performs a feature ranking based on Fvalue scores.
        feature_cumulative_ranking(): generates a cumulative ranking based on previously ranked features.
        feature_filtering(): uses a target importance to filter previously ranked features.
        feature_selection(): performs a feature selection based on a given strategy and a target importance.
    #Note
        For the feature selection, we are going to be using, among others, a PCA-based methodology. PCA is generally used as a way to reduce dimensionality, generating principal components that represent linear combinations of the original features. Using them in further analysis allows for more efficiency but the original meaning of the features is lost. In this specific case, we are interested in mantaining the original features. Therefore, we will use the PCA as a way to know which of these features contribute greatly to the overall variance of our dataset.
        One thing that we can notice is that the cumulative variance after the PCA analysis is not that different from the cumulative variance of the control case where all variables are equaly important. This comes to show that, in our dataset, all variable contribute similarly to the overall variance. Therefore, no dimensionality reduction is going to make much a difference.
    """
   
    def pca_feature_ranking(self,X):
        """
        Function that performs a feature ranking based on PCA.
        :param X: (Dataframe) dataframe with the predictive variables.
        """
        
        #The objects that are going to be created are the following:
            #variance: vector with the variance associated to each principal component.
            #standarised_variance: vector with the variances suming up to 1.
            #loadings: matrix with the variance of each variable in each principal component.
            #absolute_loadings: matrix with the loadings in absolute value.
            #standarised_loadings: matrix with the loadings in a principal component suming up to 1.
            #weighted_loadings: matrix with the variance of each variable subject to the variance of the principal component.
            #weighted_loadings_per_variable: vector with the combined variance of each variable.
            #feature_selection: dataframe with the variables ordered by their variance.
        
        num_components = len(X.columns)
        scaled_X = pandas.DataFrame(MinMaxScaler().fit_transform(X),columns=X.columns)
        pca = PCA(n_components=num_components).fit(scaled_X)
        variance = pca.explained_variance_ratio_.reshape(num_components,1)
        standarised_variance = numpy.divide(variance,numpy.sum(variance))
        loadings = pca.components_
        absolute_loadings = numpy.abs(loadings)
        standarised_loadings = numpy.divide(absolute_loadings,numpy.sum(absolute_loadings,axis=1).reshape(num_components,1))
        weighted_loadings = numpy.multiply(standarised_loadings,standarised_variance)
        weighted_loadings_per_variable = numpy.sum(weighted_loadings,axis=0)
        feature_ranking = pandas.DataFrame({'Variables':X.columns,'Importance':weighted_loadings_per_variable})
        feature_ranking = feature_ranking.sort_values(by=['Importance'],ascending=False).reset_index(drop=True)
        return feature_ranking
    
    def chi_feature_ranking(self,X,Y):
        """
        Function that performs a feature ranking based on chi scores.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        """
        scaled_X = pandas.DataFrame(MinMaxScaler().fit_transform(X),columns=X.columns)
        feature_scores = chi2(scaled_X,Y)[0]
        feature_scores = numpy.divide(feature_scores,numpy.sum(feature_scores))
        feature_scores = pandas.DataFrame({'Variables':X.columns,'Importance':feature_scores})
        feature_ranking = feature_scores.sort_values(by=['Importance'],ascending=False).reset_index(drop=True)
        return(feature_ranking)
    
    def anova_feature_ranking(self,X,Y):
        """
        Function that performs a feature ranking based on Fvalue scores.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        """
        scaled_X = pandas.DataFrame(MinMaxScaler().fit_transform(X),columns=X.columns)
        feature_scores = f_classif(scaled_X,numpy.array(Y).reshape(Y.shape[0],))[0]
        feature_scores = numpy.divide(feature_scores,numpy.sum(feature_scores))
        feature_scores = pandas.DataFrame({'Variables':X.columns,'Importance':feature_scores})
        feature_ranking = feature_scores.sort_values(by=['Importance'],ascending=False).reset_index(drop=True)
        return(feature_ranking)
        
    def feature_cumulative_ranking(self,ranked_features):
        """
        Function that generates a cumulative ranking based on previously ranked features.
        :param ranked_features: (Dataframe) dataframe with all the variables that have been ranked, ordered in decline.
        """
        cumulative_importance = 0
        cumulative_importances = []
        for row in ranked_features.index:
            cumulative_importance += ranked_features.iloc[row,1]
            cumulative_importances.append(cumulative_importance)
        ranked_features['CumImp'] = cumulative_importances
        ranked_features = ranked_features.drop(['Importance'],axis=1)
        return ranked_features

    def feature_filtering(self,cumulative_ranked_features,target_importance):
        """
        Function that uses a target importance to filter previously ranked features.
        :param cumulative_ranked_features: (Dataframe) dataframe with all the variables that have been ranked, ordered in decline and with their cumulative values.
        :param target_importance: (Float) portion of importance that we want to keep.
        """
        filtered_features = []
        for row in cumulative_ranked_features.index:
            filtered_features.append(cumulative_ranked_features.iloc[row,0])
            if cumulative_ranked_features.iloc[row,1] > target_importance: break
        return filtered_features

    def feature_seleccion(self,X,Y,selection_strategy,target_importance):
        """
        Function that performs a feature selection based on a given strategy and a target importance.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        :param selection_strategy: (String) technique to select features with higher importance.
        :param target_importance: (Float) portion of importance that we want to keep.
        """
        if selection_strategy=='pca': feature_ranking = self.pca_feature_ranking(X)
        elif selection_strategy=='chi': feature_ranking = self.chi_feature_ranking(X,Y)
        elif selection_strategy=='anova': feature_ranking = self.anova_feature_ranking(X,Y)
        feature_cumulative_ranking = self.feature_cumulative_ranking(feature_ranking)
        filtered_features = self.feature_filtering(feature_cumulative_ranking,target_importance)
        X = X[filtered_features]
        return(X)

##########################################################################################################################################

class Balancing:
    """
    Class that allows balancing a dataset.
    #Methods
        get_balanced_datasets(): selects the rows in the input and output dataset so that positive and negative values in the output are balanced.
    #Note
        We might need to consider doing cross-validation in those cases where the number of sample is very low.
    """

    def get_balanced_datasets(self,X,Y,portion_negative,minimum_size,seed):
        """
        Function that selects the rows in the input and output dataset so that positive and negative values in the output are balanced.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        :param portion_negative: (Float) portion of negative values in the output.
        :param minimum_size: (Float) portion of the orginal dataset that is considered as the minimum posible size.
        :param seed: (Integer) seed for reproducibility of random processes (if not desired, use None).
        """             
        XY = pandas.concat([X,Y],axis=1).sample(frac=1,random_state=seed).reset_index(drop=True)
        num_rows = XY.shape[0]; num_columns = XY.shape[1]; minimum_rows = round(minimum_size*num_rows)
        num_positive = list(XY.iloc[:,num_columns-1].values).count(1)
        num_negative = round((portion_negative*num_positive)/(1-portion_negative))
        num_filtered_rows = num_positive+num_negative
        #If the number of filtered rows is smaller than the minimum number of rows, we filter the dataframe and complete it with negative values.
        #If the number of filtered rows is smaller than the number of rows, we filter the dataframes.
        #If the number of filtered rows is greater than the number of rows, we do not filter the dataframes.
        if num_filtered_rows < num_rows:
            XY_positive = XY.loc[XY.iloc[:,num_columns-1]==1].iloc[0:num_positive,:]
            if num_filtered_rows < minimum_rows: XY_negative = XY.loc[XY.iloc[:,num_columns-1]==0].iloc[0:num_filtered_rows-num_positive,:]
            else: XY_negative = XY.loc[XY.iloc[:,num_columns-1]==0].iloc[0:num_negative,:]
            XY_filtered = pandas.concat([XY_positive,XY_negative]).sample(frac=1,random_state=seed).reset_index(drop=True)
            X_balanced = XY_filtered.iloc[:,0:num_columns-1]
            Y_balanced = XY_filtered.iloc[:,[num_columns-1]]
        else:
            X_balanced,Y_balanced = X,Y
        return(X_balanced,Y_balanced)

##########################################################################################################################################

class Datasets:
    """
    Class that includes different methods for the definition of the datasets with which the neural networks will work.
    #Methods
        get_train_test_datasets(): splits the input and output dataset into the train and test dataset, given their corresponding ratios.
        get_train_val_test_datasets(): splits the input and output dataset into the train, validation and test dataset, given their corresponding ratios.
        get_crossvalidated_datasets(): splits the input and output dataset into the train and test dataset, given their corresponding cross-validation index.
    #Note
        When making the splits, we should check whether they are done according to the statistic distribution of the population.
        Although we include the possibility, a validation set is not entirely necessary because there is going to be a posterior clinic validation.
    """
       
    def get_train_test_datasets(self,X,Y,train_ratio,test_ratio,input_normalization,seed):
        """
        Function that splits the input and output dataset into the train and test dataset, given their corresponding ratios.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        :param train_ratio: (Float) ratio of the whole set that will go to the train set.
        :param test_ratio: (Float) ratio of the whole set that will go to the test set.
        :param input_normalization: (Boolean) whether to normalize or not the input dataframes.
        :param seed: (Integer) seed for reproducibility of random processes (if not desired, use None).
        """
        num_patients,num_features = X.shape
        real_portion_negative = list(Y.values).count(0)/len(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_ratio,random_state=seed)
        if input_normalization==True:
            scaler = StandardScaler()
            X_train = pandas.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_test = pandas.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
        dictionary = {'X_train':X_train,'X_test':X_test,'Y_train':Y_train,'Y_test':Y_test}
        return(dictionary,num_patients,num_features,real_portion_negative)
    
    def get_train_val_test_datasets(self,X,Y,train_ratio,val_ratio,test_ratio,input_normalization,seed):
        """
        Function that splits the input and output dataset into the train, validation and test dataset, given their corresponding ratios.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        :param train_ratio: (Float) ratio of the whole set that will go to the train set.
        :param val_ratio: (Float) ratio of the whole set that will go to the validation set.
        :param test_ratio: (Float) ratio of the whole set that will go to the test set.
        :param input_normalization: (Boolean) whether to normalize or not the input dataframes.
        :param seed: (Integer) seed for reproducibility of random processes (if not desired, use None).
        """
        num_patients,num_features = X.shape
        real_portion_negative = list(Y.values).count(0)/len(Y)
        X_train, X_val_test, Y_train, Y_val_test = train_test_split(X,Y,test_size=(val_ratio+test_ratio)/(train_ratio+val_ratio+test_ratio),random_state=seed)
        X_val, X_test, Y_val, Y_test = train_test_split(X_val_test,Y_val_test,test_size=test_ratio/(val_ratio+test_ratio),random_state=seed)
        if input_normalization==True:
            scaler = StandardScaler()
            X_train = pandas.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_val = pandas.DataFrame(scaler.transform(X_val),columns=X_val.columns,index=X_val.index)
            X_test = pandas.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
        dictionary = {'X_train':X_train,'X_val':X_val,'X_test':X_test,'Y_train':Y_train,'Y_val':Y_val,'Y_test':Y_test}
        return(dictionary,num_patients,num_features,real_portion_negative)

    def get_crossvalidated_datasets(self,X,Y,train_index,test_index,input_normalization):
        """
        Function that splits the input and output dataset into the train and test dataset, given their corresponding cross-validation index.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        :param train_index: (List) list with the indexes for the train data.
        :param test_index: (List) list with the indexes for the test data.
        :param input_normalization: (Boolean) whether to normalize or not the input dataframes.
        """
        num_patients,num_features = X.shape
        real_portion_negative = list(Y.values).count(0)/len(Y)
        X_train = X.iloc[train_index,:]
        Y_train = Y.iloc[train_index,:]
        X_test = X.iloc[test_index,:]
        Y_test = Y.iloc[test_index,:]    
        if input_normalization==True:
            scaler = StandardScaler()
            X_train = pandas.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns,index=X_train.index)
            X_test = pandas.DataFrame(scaler.transform(X_test),columns=X_test.columns,index=X_test.index)
        dictionary = {'X_train':X_train,'X_test':X_test,'Y_train':Y_train,'Y_test':Y_test}
        return(dictionary,num_patients,num_features,real_portion_negative)

##########################################################################################################################################

class Models:
    """
    Class that includes different methods for the definition of the model with which the neural networks will work.
    #Methods
        run_model(): builds, compiles and trains a neural network model.
        run_model_with_validation(): builds, compiles and trains a neural network model, considering validation.
        build_model(): builds the architecture of a neural network model.
        build_model_dropout(): builds the architecture of a neural network model, considering dropout layers.
        build_model_regularised(): builds the architecture of a neural network model, considering regularization.
    """

    def run_model(self,datasets,weighting,class_weights,regularization,dropout_rate,l2_rate,num_epochs,num_features,scaling_factor,output_activation):
        """
        Function that builds, compiles and trains a neural network model.
        :param datasets: (Dicctionary) dictionary with the datasets required for training. 
        :param weighting: (Boolean) whether to use or not weighting.
        :param class_weights: (Dictionary) class weights for negative and positive outputs.
        :param regularization: (String) type of regularization to use.
        :param dropout_rate: (Float) fraction of the units to drop.
        :param l2_rate: (Float) magnitude of the weight decay.
        :param num_epochs: (Integer) number of training iterations.
        :param num_features: (Integer) number of variables that the model receives.
        :param scaling_factor: (Float) factor to determine the number of layers.
        :param output_activation: (String) activation for the output layer.
        """
        #Define some parameters
        X_train = numpy.asarray(datasets['X_train'])
        Y_train = numpy.asarray(datasets['Y_train']).flatten()
        Y_train_categorical = to_categorical(Y_train)
        compiled_metric = [BinaryAccuracy(name='accuracy'),Precision(name='precision'),Recall(name='recall')]
        #Define some parameters relative to unbalancing
        negatives = list(Y_train).count(0); positives = list(Y_train).count(1); total = negatives+positives
        negatives_proportion = negatives/total; positives_proportion = positives/total
        if class_weights=='Default': class_weights = {0:positives_proportion,1:negatives_proportion}
        #Build, compile and train the model
        if output_activation=='sigmoid':
            output_bias = numpy.log([positives+1e-10/negatives])
            if regularization==None: model,hidden_units = self.build_model(num_features,scaling_factor,output_activation,output_bias)
            elif regularization=='dropout': model,hidden_units = self.build_model_dropout(num_features,scaling_factor,output_activation,output_bias,dropout_rate)
            elif regularization=='l2': model,hidden_units = self.build_model_l2(num_features,scaling_factor,output_activation,output_bias,l2_rate)
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=compiled_metric)
            if weighting==True: fit = model.fit(X_train,Y_train,epochs=num_epochs,class_weight=class_weights,verbose=0)
            elif weighting==False: fit = model.fit(X_train,Y_train,epochs=num_epochs,verbose=0)
        elif output_activation=='softmax':
            output_bias = [numpy.log(negatives_proportion),numpy.log(positives_proportion+1e-10)]
            if regularization==None: model,hidden_units = self.build_model(num_features,scaling_factor,output_activation,output_bias)
            elif regularization=='dropout': model,hidden_units = self.build_model_dropout(num_features,scaling_factor,output_activation,output_bias,dropout_rate)
            elif regularization=='l2': model,hidden_units = self.build_model_l2(num_features,scaling_factor,output_activation,output_bias,l2_rate)
            model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=compiled_metric)
            if weighting==True: fit = model.fit(X_train,Y_train_categorical,epochs=num_epochs,class_weight=class_weights,verbose=0)
            elif weighting==False: fit = model.fit(X_train,Y_train_categorical,epochs=num_epochs,verbose=0)           
        #Return the model
        return(model,hidden_units,fit)

    def run_model_with_validation(self,datasets,weighting,class_weights,regularization,dropout_rate,l2_rate,num_epochs,num_features,scaling_factor,output_activation):
        """
        Function that builds, compiles and trains a neural network model, considering validation.
        :param datasets: (Dicctionary) dictionary with the datasets required for training. 
        :param weighting: (Boolean) whether to use or not weighting.
        :param class_weights: (Dictionary) class weights for negative and positive outputs.
        :param regularization: (String) type of regularization to use.
        :param dropout_rate: (Float) fraction of the units to drop.
        :param l2_rate: (Float) magnitude of the weight decay.
        :param num_epochs: (Integer) number of training iterations.
        :param num_features: (Integer) number of variables that the model receives.
        :param scaling_factor: (Float) factor to determine the number of layers.
        :param output_activation: (String) activation for the output layer.
        """
        #Define some parameters
        X_train = numpy.asarray(datasets['X_train'])
        X_val = numpy.asarray(datasets['X_val'])
        Y_train = numpy.asarray(datasets['Y_train']).flatten()
        Y_train_categorical = to_categorical(Y_train)
        Y_val = numpy.asarray(datasets['Y_val']).flatten()
        Y_val_categorical = to_categorical(Y_val)
        compiled_metric = [BinaryAccuracy(name='accuracy'),Precision(name='precision'),Recall(name='recall')]
        #Define some parameters relative to unbalancing
        negatives = list(Y_train).count(0); positives = list(Y_train).count(1); total = negatives+positives
        negatives_proportion = negatives/total; positives_proportion = positives/total
        if class_weights=='Default': class_weights = {0:positives_proportion,1:negatives_proportion}
        #Define early stopping
        early_stopping = EarlyStopping(monitor='val_precision',mode='max',patience=num_epochs,restore_best_weights=True)
        #Build, compile and train the model
        if output_activation=='sigmoid':
            output_bias = numpy.log([positives+1e-10/negatives])
            if regularization==None: model,hidden_units = self.build_model(num_features,scaling_factor,output_activation,output_bias)
            elif regularization=='dropout': model,hidden_units = self.build_model_dropout(num_features,scaling_factor,output_activation,output_bias,dropout_rate)
            elif regularization=='l2': model,hidden_units = self.build_model_l2(num_features,scaling_factor,output_activation,output_bias,l2_rate)
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=compiled_metric)
            if weighting==True: fit = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),epochs=num_epochs,class_weight=class_weights,callbacks=[early_stopping],verbose=0)
            elif weighting==False: fit = model.fit(X_train,Y_train,validation_data=(X_val,Y_val),epochs=num_epochs,callbacks=[early_stopping],verbose=0)
        elif output_activation=='softmax':
            output_bias = [numpy.log(negatives_proportion),numpy.log(positives_proportion+1e-10)]
            if regularization==None: model,hidden_units = self.build_model(num_features,scaling_factor,output_activation,output_bias)
            elif regularization=='dropout': model,hidden_units = self.build_model_dropout(num_features,scaling_factor,output_activation,output_bias,dropout_rate)
            elif regularization=='l2': model,hidden_units = self.build_model_l2(num_features,scaling_factor,output_activation,output_bias,l2_rate)
            model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=compiled_metric)
            if weighting==True: fit = model.fit(X_train,Y_train_categorical,validation_data=(X_val,Y_val_categorical),epochs=num_epochs,class_weight=class_weights,callbacks=[early_stopping],verbose=0)
            elif weighting==False: fit = model.fit(X_train,Y_train_categorical,validation_data=(X_val,Y_val_categorical),epochs=num_epochs,callbacks=[early_stopping],verbose=0)           
        #Return the model
        return(model,hidden_units,fit)

    def build_model(self,num_features,scaling_factor,output_activation,output_bias):
        """
        Function that builds the architecture of a neural network model.
        :param num_features: (Integer) number of variables that the model receives.
        :param scaling_factor: (Float) factor to determine the number of layers.
        :param output_activation: (String) activation for the output layer.
        :param output_bias: (Float) initial bias for the output activation.
        """
        #hidden_units = 20
        #hidden_units = round(num_training_samples/((num_features+1)*scaling_factor))
        hidden_units = round(scaling_factor*(sqrt(num_features+1)))
        inputs = Input(shape=(num_features,),name='inputs')
        hidden_1 = Dense(hidden_units,activation='relu',kernel_initializer='he_normal',name='hidden_1')(inputs)
        hidden_2 = Dense(hidden_units,activation='relu',kernel_initializer='he_normal',name='hidden_2')(hidden_1)
        if output_activation=='sigmoid': outputs = Dense(1,activation='sigmoid',kernel_initializer='glorot_normal',bias_initializer=Constant(output_bias),name='output')(hidden_2)
        elif output_activation=='softmax': outputs = Dense(2,activation='softmax',kernel_initializer='glorot_normal',bias_initializer=Constant(output_bias),name='output')(hidden_2)
        model = Model(inputs=inputs, outputs=outputs)
        return(model,hidden_units)
    
    def build_model_dropout(self,num_features,scaling_factor,output_activation,output_bias,dropout_rate):
        """
        Function that builds the architecture of a neural network model, considering dropout layers.
        :param num_features: (Integer) number of variables that the model receives.
        :param scaling_factor: (Float) factor to determine the number of layers.
        :param output_activation: (String) activation for the output layer.
        :param output_bias: (Float) initial bias for the output activation.
        :param dropout_rate: (Float) fraction of the hidden units to drop.
        """
        #hidden_units = 20
        #hidden_units = round(num_training_samples/((num_features+1)*scaling_factor))
        hidden_units = round(scaling_factor*(sqrt(num_features+1)))
        inputs = Input(shape=(num_features,),name='inputs')
        hidden_1 = Dense(hidden_units,activation='relu',kernel_initializer='he_normal',name='hidden_1')(inputs)
        dropout_1 = Dropout(rate=dropout_rate)(hidden_1)
        hidden_2 = Dense(hidden_units,activation='relu',kernel_initializer='he_normal',name='hidden_2')(dropout_1)
        dropout_2 = Dropout(rate=dropout_rate)(hidden_2)
        if output_activation=='sigmoid': outputs = Dense(1,activation='sigmoid',kernel_initializer='glorot_normal',bias_initializer=Constant(output_bias),name='output')(dropout_2)
        elif output_activation=='softmax': outputs = Dense(2,activation='softmax',kernel_initializer='glorot_normal',bias_initializer=Constant(output_bias),name='output')(dropout_2)
        model = Model(inputs=inputs, outputs=outputs)
        return(model,hidden_units)

    def build_model_l2(self,num_features,scaling_factor,output_activation,output_bias,l2_rate):
        """
        Function that builds the architecture of a neural network model, considering regularization.
        :param num_features: (Integer) number of variables that the model receives.
        :param scaling_factor: (Float) factor to determine the number of layers.
        :param output_activation: (String) activation for the output layer.
        :param output_bias: (Float) initial bias for the output activation.
        :param l2_rate: (Float) magnitude of the weight decay.
        """
        #hidden_units = 20
        #hidden_units = round(num_training_samples/((num_features+1)*scaling_factor))
        hidden_units = round(scaling_factor*(sqrt(num_features+1)))
        inputs = Input(shape=(num_features,),name='inputs')
        hidden_1 = Dense(hidden_units,activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(l2_rate),name='hidden_1')(inputs)
        hidden_2 = Dense(hidden_units,activation='relu',kernel_initializer='he_normal',kernel_regularizer=l2(l2_rate),name='hidden_2')(hidden_1)
        if output_activation=='sigmoid': outputs = Dense(1,activation='sigmoid',kernel_initializer='glorot_normal',bias_initializer=Constant(output_bias),name='output')(hidden_2)
        elif output_activation=='softmax': outputs = Dense(2,activation='softmax',kernel_initializer='glorot_normal',bias_initializer=Constant(output_bias),name='output')(hidden_2)
        model = Model(inputs=inputs, outputs=outputs)
        return(model,hidden_units)
      
##########################################################################################################################################

class History:
    """
    Class that includes different methods for the visualitation of the training history.
    #Methods
        get_history(): collects the fit history.
        get_history_plot(): generates a plot with the fit history of a given metric.
    """

    def get_history(self,fit,validation):
        """
        Function that collects the fit history.
        :param fit: (Object) object that stores the train information.
        :param validation: (Boolean) whether to record validation history.
        """
        dictionary,train_dictionary,val_dictionary={},{},{}
        keys = list(fit.history.keys())
        for metric in ['loss','accuracy','precision','recall']:
            for key in keys:
                if re.search('val_'+str(metric),key):
                    val_dictionary[str(metric)] = numpy.array(fit.history[key])
                elif re.search(str(metric),key):
                    train_dictionary[str(metric)] = numpy.array(fit.history[key])
        dictionary['train'] = train_dictionary
        if validation==True: dictionary['val'] = val_dictionary
        return(dictionary)
                
    def get_history_plot(self,metric,model,train,validation,val):
        """
        Function that generates a plot with the fit history of a given metric.
        :param metric: (String) metric to be studied.
        :param model: (String) identifier for the model.
        :param train: (List) list with the train history of a given metric.
        :param validation: (Boolean) whether to record validation history. 
        :param val: (List) list with the validation history of a given metric.
        """
        figure = matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(train)
        if validation==True: matplotlib.pyplot.plot(val)
        matplotlib.pyplot.title('Evolution of '+str(metric)+' ('+model+')')
        matplotlib.pyplot.ylabel(metric)
        matplotlib.pyplot.xlabel('epochs')
        matplotlib.pyplot.legend(['Train', 'Validation'], loc='upper left')
        return(figure)
   
##########################################################################################################################################

class Results:
    """
    Class that includes different methods for retrieving the results.
    #Methods
        get_results(): stores the predicted outputs and compares then with the real outputs, giving the corresponding metrics and confusion data.
    """
    
    def get_results(self,predict,real_output,output_activation):
        """
        Function that stores the predicted outputs and compares then with the real outputs, giving the corresponding metrics and confusion data.
        :param predict: (Array) predicted values for the output.
        :param real_output: (Dataframe) real values for the output.
        :param output_activation: (String) activation for the output layer.
        """
        if output_activation=='sigmoid': predicted_output = pandas.DataFrame(numpy.rint(predict)).astype('int')
        elif output_activation=='softmax': predicted_output = pandas.DataFrame(numpy.rint(predict[:,1])).astype('int')
        predicted_output.columns = real_output.columns
        predicted_output.index = real_output.index
        comparison = predicted_output.apply(lambda col: col.astype(str)) + real_output.apply(lambda col: col.astype(str))
        comparison = comparison.replace({'11':'TP','00':'TN','10':'FP','01':'FN'})
        comp_list = list(comparison.values.flatten())
        #True Positive (TP): the real and predicted values are positive.
        #True Negative (TN): the real and predicted values are negative.
        #False Positive (FP): the predicted value is positive but the real value is negative.
        #False Negative (FN): the predicted value is negative but the real value is positive.
        TP = comp_list.count('TP'); TN = comp_list.count('TN'); FP = comp_list.count('FP'); FN = comp_list.count('FN')
        #Accuracy: values correctly predicted among the total values (how often the predictions are true).
        #Precision: correctly predicted values among the predicted positive values (how often positive predictions are true).
        #Recall: correctly predicted values among the real positive values (how often positive values are predicted as such).
        #F1: harmonic mean between precision and recall.
        accuracy = (TP+TN)/(TP+TN+FP+FN+1e-10); precision = TP/(TP+FP+1e-10); recall = TP/(TP+FN+1e-10); f1 = 2*(precision*recall)/(precision+recall+1e-10)
        confusion = pandas.DataFrame({'TP':[TP],'TN':[TN],'FP':[FP],'FN':[FN],'accuracy':[accuracy],'precision':[precision],'recall':[recall],'f1':[f1]})
        dictionary = {'predicted_output':predicted_output,'real_output':real_output,'comparison':comparison,'confusion':confusion}
        return(dictionary)

##########################################################################################################################################

class NeuralNetwork(Binarization,Imputation,FeatureSelection,Balancing,Datasets,Models,History,Results):
    """
    Class that allows running a neural network and recording the results.
    Methods:
        run_nn(): builds, trains, validates and evaluates a neural network model.
        record_nn(): records the results of the previously run neural network model.
        run_nn_cv(): builds, trains, validates and evaluates a neural network model through cross-validation.
        record_nn_cv(): records the results of the previously run neural network model with cross-validation.
        keep_best(): keeps the best model for each brain region, given an optimization criteria.
    """  

    def run_nn(self,X,Y,
                          binary_threshold=30,
                          imputation_strategy='median',
                          selection_strategy='anova',
                          target_importance=1,
                          balancing=False,
                          portion_negative=None,
                          minimum_size=None,
                          train_ratio=0.8,
                          validation=False,
                          val_ratio=None,
                          test_ratio=0.2,
                          input_normalization=True,
                          num_epochs=100,
                          scaling_factor=1,
                          output_activation='sigmoid',
                          weighting=True,
                          class_weights='Default',
                          regularization=None,
                          dropout_rate=None,
                          l2_rate=None,
                          seed=None):
        """
        Function that builds, trains, validates and evaluates a neural network model.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        :param binary_threshold: (Integer) threshold above which a value is considered to be positive.
        :param imputation_strategy: (String) technique to fill the missing values.
        :param selection_strategy: (String) technique to select features with higher importance.
        :param target_importance: (Float) portion of importance that we want to keep.
        :param balancing: (Boolean) whether to consider or not balancing.
        :param portion_negative: (Float) portion of negative values in the output.
        :param minimum_size: (Float) portion of the orginal dataset that is considered as the minimum posible size.
        :param train_ratio: (Float) ratio of the whole set that will go to the train set.
        :param validation: (Boolean) whether to consider validation data.
        :param val_ratio: (Float) ratio of the whole set that will go to the validation set.
        :param test_ratio: (Float) ratio of the whole set that will go to the test set.
        :param input_normalization: (Boolean) whether to normalize or not the input dataframes.
        :param num_epochs: (Float) number of training iterations.
        :param scaling_factor: (Float) factor to determine the number of layers.
        :param output_activation: (String) activation for the output layer.
        :param weighting: (Boolean) whether to use or not weighting.
        :param class_weights: (Dictionary) class weights for negative and positive outputs.
        :param regularization: (String) type of regularization to use.
        :param dropout_rate: (Float) fraction of the hidden units to drop.
        :param l2_rate: (Float) magnitude of the weight decay.
        :param seed: (Integer) seed for reproducibility of random processes (if not desired, use None).
        """                   
        #1. DEFINE THE STORAGE STRUCTURES
        datasets,model,history,results,parameters={},{},{},{},{}
        #2. CONVERT THE OUTPUT DATASET TO BINARY
        Y_binary = self.get_binary(Y,binary_threshold)
        #3. IMPUTATE THE INPUT DATASET
        X_imputed = self.imputate(X,imputation_strategy)
        #4. FILTER THE INPUT DATASET
        X_selected = self.feature_seleccion(X_imputed,Y_binary,selection_strategy,target_importance)
        #5. BALANCE THE DATASET
        if balancing==True: X_balanced,Y_balanced = self.get_balanced_datasets(X_selected,Y_binary,portion_negative,minimum_size,seed)
        #6. DEFINE THE TRAIN-TEST DATASETS
        if balancing==True: X_split,Y_split = X_balanced.copy(),Y_balanced.copy()
        elif balancing==False: X_split,Y_split = X_selected.copy(),Y_binary.copy()
        if validation==True: datasets,num_patients,num_features,real_portion_negative = self.get_train_val_test_datasets(X_split,Y_split,train_ratio,val_ratio,test_ratio,input_normalization,seed)
        elif validation==False: datasets,num_patients,num_features,real_portion_negative = self.get_train_test_datasets(X_split,Y_split,train_ratio,test_ratio,input_normalization,seed)
        #7. RUN THE MODEL
        if validation==True: model,hidden_units,fit = self.run_model_with_validation(datasets,weighting,class_weights,regularization,dropout_rate,l2_rate,num_epochs,num_features,scaling_factor,output_activation)
        elif validation==False: model,hidden_units,fit = self.run_model(datasets,weighting,class_weights,regularization,dropout_rate,l2_rate,num_epochs,num_features,scaling_factor,output_activation)
        #8. RECORD THE HISTORY
        history = self.get_history(fit,validation)
        #9. RECORD THE RESULTS
        predict_train = model.predict(numpy.asarray(datasets['X_train']),verbose=0)
        predict_test = model.predict(numpy.asarray(datasets['X_test']),verbose=0)
        if validation==True: predict_val = model.predict(numpy.asarray(datasets['X_val']),verbose=0)
        results['train']= self.get_results(predict_train,datasets['Y_train'],output_activation)
        results['test']= self.get_results(predict_test,datasets['Y_test'],output_activation)
        if validation==True: results['val'] = self.get_results(predict_val,datasets['Y_val'],output_activation)
        #10. RECORD THE PARAMETERS
        parameters = pandas.DataFrame({'Imputation':[imputation_strategy],
                                       'Selection':[selection_strategy],
                                       'Target importance':[target_importance],
                                       'Features':[num_features],
                                       'Balancing':[balancing],
                                       'Negatives':[real_portion_negative],
                                       'Minimum size':[minimum_size],
                                       'Patients':[num_patients],
                                       'Input normalisation':[input_normalization],                                      
                                       'Epochs':[num_epochs],
                                       'Scaling factor':[scaling_factor],
                                       'Hidden units':[hidden_units],
                                       'Output activation':[output_activation],
                                       'Weighting':[weighting],
                                       'Class weights':[class_weights],
                                       'Regularization':[regularization],
                                       'Dropout rate':[dropout_rate],
                                       'L2 rate':[l2_rate]})
        #11. RECORD THE INFORMATION
        dictionary = {'datasets':datasets,'model':model,'history':history,'results':results,'parameters':parameters}
        return(dictionary)
        """
        THIS FUNCTION RUNS A SINGLE MODEL
        1.  DEFINE THE STORAGE STRUCTURES
            The dictionaries where the information will be stored are defined.
        2.  CONVERT THE OUTPUT DATASET TO BINARY
            The output dataset is converted from quantitative to binary based on the binary threshold.
        3.  IMPUTATE THE INPUT DATASET
            The input dataset is imputed based on the imputation strategy.
            This imputation is thought for datasets with low percentages of missing values, instead, it might require from previously removing some rows or columns.
        4.  FILTER THE INPUT DATASET
            The dataset features are selected based on the feature selection strategy and the target importance (the higher the importance, the more features are selected).
        5.  BALANCE THE DATASETS
            The datasets are balanced in order to have the similar number of samples with positive and negative output, this is done by removing negative samples.
            There is a minimum size that the balanced dataset must have (if they don't reach it, they are unbalanced).
            Reproducibility might be considered through seed definition.
        6.  DEFINE THE TRAIN-TEST DATASETS
            The train and test datasets are defined.
            Input normalisation might be considered (normalisation is done in training, the mean and variance are stored and used for test -to mimic production situations-)
            Reproducibility might be considered through seed definition.
        7.  RUN THE MODEL
            The neural network is built, compiled and trained based on the output activation, scaling factor and number of epochs.
            Bias initialization is considered.
            Class weighting might be considered.
            Regularization might be considered.
        8.  RECORD THE HISTORY
            The history for the training set is recorded.
        9.  RECORD THE RESULTS
            The results for both the training and test set are recorded.
            This includes the real sets, the predicted sets and the comparison between both (useful to look at specific errors).
            This includes TP,TN,FP,FN,Accuracy,Precision,Recall,F1 (useful to look at general trends in errors).
        10. RECORD THE PARAMETERS
            The parameters that were randomly chosen are recorded (useful for determining which values work best)
        11. RECORD THE INFORMATION
            All the information is stored in a dictionary structure.
        """
    
    def record_nn(self,dictionary,key,file): 
        """
        Function that records the results of the previously run neural network model.
        :param dictionary: (Dictionary) dictionary with all the information recorded from the neural network model.
        :param key: (String) name that is given to the model.
        :param file: (String) name that is given to the file where the results are stored.
        """           
        train = dictionary['results']['train']['confusion']; train['Model'] = key
        test = dictionary['results']['test']['confusion']; test['Model'] = key
        parameters = dictionary['parameters']; parameters['Model'] = key
        train.to_csv(str(file)+'_train.csv',mode='a',index=False,header=False)
        test.to_csv(str(file)+'_test.csv',mode='a',index=False,header=False)
        parameters.to_csv(str(file)+'_parameters.csv',mode='a',index=False,header=False)

    def run_nn_cv(self,X,Y,
                             binary_threshold=30,
                             imputation_strategy='median',
                             selection_strategy='anova',
                             target_importance=1,
                             balancing=False,
                             portion_negative=None,
                             minimum_size=None,
                             k_folds=5,
                             input_normalization=True,
                             num_epochs=100,
                             scaling_factor=1,
                             output_activation='sigmoid',
                             weighting=True,
                             class_weights='Default',
                             regularization=None,
                             dropout_rate=None,
                             l2_rate=None,
                             seed=None):
        """
        Function that builds, trains, validates and evaluates a neural network model through cross-validation.
        :param X: (Dataframe) dataframe with the predictive variables.
        :param Y: (Dataframe) dataframe with the predicted variables.
        :param binary_threshold: (Integer) threshold above which a value is considered to be positive.
        :param imputation_strategy: (String) technique to fill the missing values.
        :param selection_strategy: (String) technique to select features with higher importance.
        :param target_importance: (Float) portion of importance that we want to keep.
        :param balancing: (Boolean) whether to consider or not balancing.
        :param portion_negative: (Float) portion of negative values in the output.
        :param minimum_size: (Float) portion of the orginal dataset that is considered as the minimum posible size.
        :param k_folds: (Integer) number of folds to consider for cross-validation.
        :param input_normalization: (Boolean) whether to normalize or not the input dataframes.
        :param num_epochs: (Float) number of training iterations.
        :param scaling_factor: (Float) factor to determine the number of layers.
        :param output_activation: (String) activation for the output layer.
        :param weighting: (Boolean) whether to use or not weighting.
        :param class_weights: (Dictionary) class weights for negative and positive outputs.
        :param regularization: (String) type of regularization to use.
        :param dropout_rate: (Float) fraction of the hidden units to drop.
        :param l2_rate: (Float) magnitude of the weight decay.
        :param seed: (Integer) seed for reproducibility of random processes (if not desired, use None).
        """                   
        #1. DEFINE THE STORAGE STRUCTURES
        dictionary_cv = {}
        #2. CONVERT THE OUTPUT DATASET TO BINARY
        Y_binary = self.get_binary(Y,binary_threshold)
        #3. IMPUTATE THE INPUT DATASET
        X_imputed = self.imputate(X,imputation_strategy)
        #4. FILTER THE INPUT DATASET
        X_selected = self.feature_seleccion(X_imputed,Y_binary,selection_strategy,target_importance)
        #5. BALANCE THE DATASET
        if balancing==True: X_balanced,Y_balanced = self.get_balanced_datasets(X_selected,Y_binary,portion_negative,minimum_size,seed)
        #6. DEFINE THE DATASETS OVER WHICH THE CROSS-VALIDATION WILL BE DONE
        if balancing==True: X_split,Y_split = X_balanced.copy(),Y_balanced.copy()
        elif balancing==False: X_split,Y_split = X_selected.copy(),Y_binary.copy()
        #7. ITERATE OVER EACH FOLD
        kf = KFold(n_splits=k_folds,shuffle=True,random_state=seed)
        k=0
        for train_index, test_index in kf.split(X_split):
            k+=1
            print('Fold number'+str(k))
            datasets,model,history,results,parameters={},{},{},{},{}
            #8. DEFINE THE TRAIN-TEST DATASET
            datasets,num_patients,num_features,real_portion_negative = self.get_crossvalidated_datasets(X_split,Y_split,train_index,test_index,input_normalization)
            #9. RUN THE MODEL
            model,hidden_units,fit = self.run_model(datasets,weighting,class_weights,regularization,dropout_rate,l2_rate,num_epochs,num_features,scaling_factor,output_activation)
            #10. RECORD THE HISTORY
            history = self.get_history(fit,validation=False)
            #11. RECORD THE RESULTS
            predict_train = model.predict(numpy.asarray(datasets['X_train']),verbose=0)
            predict_test = model.predict(numpy.asarray(datasets['X_test']),verbose=0)
            results['train']= self.get_results(predict_train,datasets['Y_train'],output_activation)
            results['test']= self.get_results(predict_test,datasets['Y_test'],output_activation)
            #12. RECORD THE PARAMETERS
            parameters = pandas.DataFrame({'Imputation':[imputation_strategy],
                                           'Selection':[selection_strategy],
                                           'Target importance':[target_importance],
                                           'Features':[num_features],
                                           'Balancing':[balancing],
                                           'Negatives':[real_portion_negative],
                                           'Minimum size':[minimum_size],
                                           'Patients':[num_patients],
                                           'Input normalisation':[input_normalization],
                                           'Epochs':[num_epochs],
                                           'Scaling factor':[scaling_factor],
                                           'Hidden units':[hidden_units],
                                           'Output activation':[output_activation],
                                           'Weighting':[weighting],
                                           'Class weights':[class_weights],
                                           'Regularization':[regularization],
                                           'Dropout rate':[dropout_rate],
                                           'L2 rate':[l2_rate]})
            #13. RECORD THE INFORMATION FOR THE FOLD
            dictionary = {'datasets':datasets,'model':model,'history':history,'results':results,'parameters':parameters}
            dictionary_cv[str(k)] = dictionary
            del dictionary
        #14. RECORD THE INFORMATION FOR ALL THE FOLDS
        return(dictionary_cv)

    def record_nn_cv(self,dictionary_cv,k_folds,key,file): 
        """
        Function that records the results of the previously run neural network model with cross-validation.
        :param dictionary_cv: (Dictionary) dictionary with all the information recorded from the neural network model with cross-validation.
        :param k_folds: (Integer) number of folds to consider for cross-validation.
        :param key: (String) name that is given to the model.
        :param file: (String) name that is given to the file where the results are stored.
        """   
        for k in range(1,k_folds+1):
            temp_train = dictionary_cv[str(k)]['results']['train']['confusion']
            temp_test = dictionary_cv[str(k)]['results']['test']['confusion']
            try: train = pandas.concat([train,temp_train],axis=0)
            except: train = temp_train
            try: test = pandas.concat([test,temp_test],axis=0)
            except: test = temp_test
        train = pandas.concat([train.mean().add_suffix('_mean'),train.std().add_suffix('_std')],axis=0).to_frame().transpose(); train['Model'] = key
        test = pandas.concat([test.mean().add_suffix('_mean'),test.std().add_suffix('_std')],axis=0).to_frame().transpose(); test['Model'] = key
        parameters = dictionary_cv[str(k)]['parameters']; parameters['Model'] = key
        train.to_csv(str(file)+'_train.csv',mode='a',index=False,header=False)
        test.to_csv(str(file)+'_test.csv',mode='a',index=False,header=False)
        parameters.to_csv(str(file)+'_parameters.csv',mode='a',index=False,header=False)

    def keep_best(self,train,test,parameters,regions,criteria,file):
        """
        Function that keeps the best model for each brain region, given an optimization criteria.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model. 
        :param parameters: (Dataframe) dataframe with the parameters information for each model.
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
                temp_df = df.loc[df['Model'].str.contains('^'+str(region)+'_')]
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
        best_parameters = parameters.loc[parameters['Model'].isin(best_models)]
        best_parameters['Model'] = pandas.Categorical(best_parameters['Model'],categories=best_models,ordered=True)
        best_parameters = best_parameters.sort_values('Model')
        best_parameters.to_csv(str(file)+'_best_parameters.csv',mode='a',index=False)
    
##########################################################################################################################################

class Evaluation:
    """
    Class that includes different methods for evaluating the results.
    #Methods
        evaluate_with_plots(): evaluates with plots the performance of models with a given value of a parameter.
        evaluate_with_plots_cv(): evaluates with plots the performance of cross-validated optimized models.
        evaluate_with_tables(): evaluates with tables the performance of models with a given value of a parameter.
        evaluate_with_tables_cv(): evaluates with tables the performance of cross-validated optimized models.
        define_df(): defines the metrics and confusion dataframes.
        define_df_cv(): defines the metrics and confusion dataframes from cross-validated optimized models.
        filter_df(): filters a dataframe based on a given value of a parameter.
        stats_df(): retrieves the statistics of a given dataframe.
        plot_df(): plots the metrics and confusion dataframes.
    """
          
    def evaluate_with_plots(self,parameters,train,test,list_parameters,list_values,region='all',additional_parameter=None):
        """
        Function that evaluates with plots the performance of models with a given value of a parameter.
        :param parameters: (Dataframe) dataframe with the parameters information for each model.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model.
        :param list_parameters: (List) parameters with which the filtering is done.
        :param list_values: (List) values of the parameters that are filtered.
        :param region: (String) region, if specified, that will be filtered.
        :param additional_parameter: (String) additional parameter from which we want information.
        """    
        metrics,train_confusion,test_confusion = self.define_df(parameters,train,test)
        filtered_metrics = self.filter_df(metrics,list_parameters,list_values,region,parameters)
        filtered_train_confusion = self.filter_df(train_confusion,list_parameters,list_values,region,parameters)
        filtered_test_confusion = self.filter_df(test_confusion,list_parameters,list_values,region,parameters)
        stats_metrics = self.stats_df(filtered_metrics,'metrics')
        stats_train_confusion = self.stats_df(filtered_train_confusion,'confusion')
        stats_test_confusion = self.stats_df(filtered_test_confusion,'confusion')
        if additional_parameter!=None: additional_information = self.add_information(parameters,list_parameters,list_values,region,additional_parameter)
        else: additional_information = None
        self.plot_df(stats_metrics,stats_train_confusion,stats_test_confusion,list_parameters,list_values,region,additional_information)

    def evaluate_with_plots_cv(self,train,test,region):
        """
        Function that evaluates with plots the performance of cross-validated optimized models.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model.
        :param region: (String) region, if specified, that will be filtered.
        """
        metrics,train_confusion,test_confusion = self.define_df_cv(train,test,region)
        self.plot_df(metrics,train_confusion,test_confusion,None,None,region,None)
        
    def evaluate_with_tables(self,parameters,train,test,list_parameters,list_values,region='all',additional_parameter=None):
        """
        Function that evaluates with tables the performance of models with a given value of a parameter.
        :param parameters: (Dataframe) dataframe with the parameters information for each model.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model.
        :param list_parameters: (List) parameters with which the filtering is done.
        :param list_values: (List) values of the parameters that are filtered.
        :param region: (String) region, if specified, that will be filtered.
        :param additional_parameter: (String) additional parameter from which we want information.
        """
        #Define the title of the table
        if additional_parameter!=None: additional_information = self.add_information(parameters,list_parameters,list_values,region,additional_parameter)
        else: additional_information = None
        string = ''
        if list_parameters!=None and list_values!=None: 
            for parameter,value in zip(list_parameters,list_values): string += str(parameter)+'='+str(value)+', '
        if additional_information!=None: string += str(additional_information)+', '
        if region!='all': string += 'Region='+str(region)+', '
        string = string.rstrip(', ')
        #Define the table
        metrics,train_confusion,test_confusion = self.define_df(parameters,train,test)
        filtered_metrics = self.filter_df(metrics,list_parameters,list_values,region,parameters)
        stats_metrics = self.stats_df(filtered_metrics,'metrics')
        float_to_str = lambda flt: str(flt).ljust(5,"0")
        table = round(stats_metrics['Mean'],3).map(float_to_str) + '+/-' + round(stats_metrics['Std'],3).map(float_to_str)
        table = table.to_frame()
        table.columns = [string]
        return(table)

    def evaluate_with_tables_cv(self,train,test,region):
        """
        Function that evaluates with tables the performance of cross-validated optimized models.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model.
        :param region: (String) region, if specified, that will be filtered.
        """
        train = train.loc[train['Model'].str.contains('^'+str(region)+'_')]
        test = test.loc[test['Model'].str.contains('^'+str(region)+'_')]
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
        table = round(stats_metrics['Mean'],3).map(float_to_str) + '+/-' + round(stats_metrics['Std'],3).map(float_to_str)
        table = table.to_frame()
        table.columns = ['Region='+str(region)]
        return(table)
        
    def define_df(self,parameters,train,test):
        """
        Function that defines the metrics and confusion dataframes.
        :param parameters: (Dataframe) dataframe with the parameters information for each model.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model.
        """
        train_metrics = pandas.concat([train[['accuracy','precision','recall','f1']].add_suffix('_train'),train[['Model']]],axis=1)
        test_metrics = pandas.concat([test[['accuracy','precision','recall','f1']].add_suffix('_test'),test[['Model']]],axis=1)
        metrics = parameters.merge(train_metrics,on='Model').merge(test_metrics,on='Model').sort_index(axis=1)
        train_confusion = parameters.merge(pandas.concat([train[['TP','TN','FP','FN']].add_suffix('_train'),train[['Model']]],axis=1),on='Model')
        test_confusion = parameters.merge(pandas.concat([test[['TP','TN','FP','FN']].add_suffix('_test'),test[['Model']]],axis=1),on='Model')
        return(metrics,train_confusion,test_confusion)

    def define_df_cv(self,train,test,region):
        """
        Function that defines the metrics and confusion dataframes from cross-validated optimized models.
        :param train: (Dataframe) dataframe with the train information for each model.
        :param test: (Dataframe) dataframe with the test information for each model.
        :param region: (String) region, if specified, that will be filtered.
        """
        train = train.loc[train['Model'].str.contains('^'+str(region)+'_')]
        test = test.loc[test['Model'].str.contains('^'+str(region)+'_')]
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
        metrics = pandas.concat([metrics_mean,metrics_std],axis=1)
        metrics.columns = ['Mean','Std']
        train_confusion = train[['TP_mean','TN_mean','FP_mean','FN_mean']]
        train_confusion.columns = ['TP_train','TN_train','FP_train','FN_train']
        train_confusion = pandas.DataFrame({'Pct':numpy.divide(train_confusion.sum(),train_confusion.values.sum())})
        test_confusion = test[['TP_mean','TN_mean','FP_mean','FN_mean']]
        test_confusion.columns = ['TP_test','TN_test','FP_test','FN_test']
        test_confusion = pandas.DataFrame({'Pct':numpy.divide(test_confusion.sum(),test_confusion.values.sum())})
        return(metrics,train_confusion,test_confusion)

    def filter_df(self,df,list_parameters,list_values,region,parameters):
        """
        Function that filters a dataframe based on a given value of a parameter.
        :param df: (Dataframe) dataframe that will be filtered.
        :param list_parameters: (List) parameters with which the filtering is done.
        :param list_values: (List) values of the parameters that are filtered.
        :param region: (String) region, if specified, that will be filtered.
        :param parameters: (Dataframe) dataframe with the parameters information for each model.
        """
        string = 'filtered_df = df.loc['
        for parameter,value in zip(list_parameters,list_values):
            if type(value)==str: string += '(df["'+str(parameter)+'"]=="'+str(value)+'")&'
            else: string += '(df["'+str(parameter)+'"]=='+str(value)+')&'
        if region!='all': string += '(df["Model"].str.contains("'+str(region)+'"))'
        string = string.rstrip('&')
        string += '].iloc[:,'+str(len(parameters.columns))+':]'
        exec(string,locals(),globals())
        return(filtered_df)
      
    def stats_df(self,df,df_type):
        """
        Function that retrieves the statistics of a given dataframe.
        :param df: (Dataframe) dataframe that was filtered.
        :param df_type: (String) type of information stored in the dataframe.
        """
        if df_type=='metrics': stats_df = pandas.DataFrame({'Mean':df.mean(),'Std':df.std()})
        elif df_type=='confusion': stats_df = pandas.DataFrame({'Pct':numpy.divide(df.sum(),df.values.sum())})
        return(stats_df)

    def add_information(self,df,list_parameters,list_values,region,additional_parameter):
        """
        Function that allows to add information about a parameter.
        :param df: (Dataframe) dataframe that will be filtered.
        :param list_parameters: (List) parameters with which the filtering is done.
        :param list_values: (List) values of the parameters that are filtered.
        :param region: (String) region, if specified, that will be filtered.
        :param additional_parameter: (String) additional parameter from which we want information.
        """
        string1 = 'filtered_df = df.loc['
        for parameter,value in zip(list_parameters,list_values):
            if type(value)==str: string1 += '(df["'+str(parameter)+'"]=="'+str(value)+'")&'
            else: string1 += '(df["'+str(parameter)+'"]=='+str(value)+')&'
        if region!='all': string += '(df["Model"].str.contains("'+str(region)+'"))'
        string1 = string1.rstrip('&')
        string1 += '][["'+str(additional_parameter)+'"]]'
        exec(string1,locals(),globals())
        string2 = str(additional_parameter)+'='+str(round(float(filtered_df.mean()),2))
        return(string2)

    def plot_df(self,stats_metrics,stats_train_confusion,stats_test_confusion,list_parameters,list_values,region,additional_information):
        """
        Function that plots the metrics and confusion dataframes.
        :param stats_metrics: (Dataframe) dataframe with statistical information from the metrics.
        :param stats_train_confusion: (Dataframe) dataframe with statistical information from the train confusion.
        :param stats_train_confusion: (Dataframe) dataframe with statistical information from the test confusion.
        :param list_parameters: (List) parameters with which the filtering was done.
        :param list_values: (List) values of the parameters that were filtered.
        :param region: (String) region, if specified, that was filtered.
        :param additional_information: (String) additional information that is added to the title.
        """
        #Define the basic organisation of the figure
        fig = matplotlib.pyplot.figure(figsize=(12,12))
        grid = matplotlib.gridspec.GridSpec(nrows=2,ncols=2,figure=fig)
        #Define the title of the figure
        string = ''
        if list_parameters!=None and list_values!=None: 
            for parameter,value in zip(list_parameters,list_values): string += str(parameter)+'='+str(value)+', '
        if additional_information!=None: string += str(additional_information)+', '
        if region!='all': string += 'Region='+str(region)+', '
        string = string.rstrip(', ')
        fig.suptitle(string,fontsize=14,y=0.925)
        #Define the first subplot with the stats_metric dataframe
        viridis = matplotlib.pyplot.cm.get_cmap('viridis',4)
        axes1 = fig.add_subplot(grid[0:1,0:2])
        counter1,counter2 = 0,0
        for metric in stats_metrics.index:
            Mean = stats_metrics['Mean'][metric]
            Std = stats_metrics['Std'][metric]
            axes1.bar(x=metric,height=Mean,yerr=Std,color=viridis(counter2),alpha=0.7,label=metric)
            axes1.text(x=counter1-0.4,y=Mean+0.01,s=round(Mean,2),size=10)
            counter1+=1
            if counter1%2==0: counter2+=1
        matplotlib.pyplot.ylim(bottom=0, top=1)
        matplotlib.pyplot.gca().spines['top'].set_visible(False)
        matplotlib.pyplot.gca().spines['right'].set_visible(False)
        axes1.set_title('Metrics',fontsize=12)
        #Define the second subplot with the stats_train_confusion dataframe
        train_confusion_matrix = numpy.array([[stats_train_confusion['Pct']['TN_train'],stats_train_confusion['Pct']['FP_train']],[stats_train_confusion['Pct']['FN_train'],stats_train_confusion['Pct']['TP_train']]])
        axes2 = fig.add_subplot(grid[1:2,0:1])
        axes2.matshow(train_confusion_matrix,cmap=matplotlib.pyplot.cm.Blues,alpha=0.7)
        for i in range(train_confusion_matrix.shape[0]):
            for j in range(train_confusion_matrix.shape[1]):
                axes2.text(x=j,y=i,s=round(train_confusion_matrix[i,j],2),va='center',ha='center',size=10)
        axes2.set(xlabel='Predicted label',ylabel='True label')
        axes2.set_title('Train confusion',fontsize=12)
        #Define the second subplot with the stats_test_confusion dataframe
        test_confusion_matrix = numpy.array([[stats_test_confusion['Pct']['TN_test'],stats_test_confusion['Pct']['FP_test']],[stats_test_confusion['Pct']['FN_test'],stats_test_confusion['Pct']['TP_test']]])
        axes3 = fig.add_subplot(grid[1:2,1:2])
        axes3.matshow(test_confusion_matrix,cmap=matplotlib.pyplot.cm.Blues,alpha=0.7)
        for i in range(test_confusion_matrix.shape[0]):
            for j in range(test_confusion_matrix.shape[1]):
                axes3.text(x=j,y=i,s=round(test_confusion_matrix[i,j],2),va='center',ha='center',size=10)
        axes3.set(xlabel='Predicted label',ylabel='True label')
        axes3.set_title('Test confusion',fontsize=12)
        #Show the figure
        matplotlib.pyplot.show()