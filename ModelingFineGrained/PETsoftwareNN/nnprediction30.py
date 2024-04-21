#Import all packages and module dependencies

import sys
sys.path.insert(0,'../../Tools')
from nntools import *
from prtools import *
nn = NeuralNetwork()
tr = Transformations()
da = Datasets()
rd = RuleDefinition()
ra = RuleApplication()

#Load the data

models = pandas.read_csv('./Optimization_30/optimization_best_parameters.csv')
X = pandas.read_csv('../../Data/X_25_diag.csv')
X_first_encoded = tr.encode_column(X,'diagnostic',{3:'FTD',5:'AD',7:'AD',13:'CT',16:'CT',17:'AD'})
X_second_encoded = tr.encode_column(X_first_encoded,'sex',{1:'male',2:'female'})
X = X_second_encoded
Y_aal_quan = pandas.read_csv('../../Data/Y_aal_quan.csv')
Y_aal_quan_diag = pandas.read_csv('../../Data/Y_aal_quan_diag.csv')
Y_brodmann_quan = pandas.read_csv('../../Data/Y_brodmann_quan.csv')
Y_brodmann_quan_diag = pandas.read_csv('../../Data/Y_brodmann_quan_diag.csv')
Y_aal_quan_AD = da.filter_diagnosis(Y_aal_quan_diag,[5,7,17])
Y_aal_quan_FTD = da.filter_diagnosis(Y_aal_quan_diag,[3])
Y_aal_quan_CT = da.filter_diagnosis(Y_aal_quan_diag,[13,16])
Y_brodmann_quan_AD = da.filter_diagnosis(Y_brodmann_quan_diag,[5,7,17])
Y_brodmann_quan_FTD = da.filter_diagnosis(Y_brodmann_quan_diag,[3])
Y_brodmann_quan_CT = da.filter_diagnosis(Y_brodmann_quan_diag,[13,16])

#Load the neural networks and predict with them
#Create a consensus prediction considering all replicas

tasks_aal = list(Y_aal_quan.columns)
tasks_brodmann = list(Y_brodmann_quan.columns)
replicas = 5
folds = 5
for replica in range(replicas):
    print(replica)
    prediction_aal = pandas.DataFrame()
    prediction_brodmann = pandas.DataFrame()
    for task in tasks_aal:
        print(task)
        try:
            Y = Y_aal_quan[[task]]
            model = models.loc[models['Model'].str.contains('^'+str(task)+'_')]
            dictionary_cv = nn.run_nn_cv(X,Y,k_folds=folds,
                                                    binary_threshold=30,
                                                    imputation_strategy=model['Imputation'].values[0],
                                                    selection_strategy=model['Selection'].values[0],
                                                    target_importance=model['Target importance'].values[0],
                                                    num_epochs=model['Epochs'].values[0],
                                                    scaling_factor=model['Scaling factor'].values[0],
                                                    output_activation=model['Output activation'].values[0],
                                                    regularization=model['Regularization'].values[0],
                                                    dropout_rate=model['Dropout rate'].values[0])
            temp_prediction = pandas.DataFrame()
            for key in dictionary_cv.keys():
                try: temp_prediction = pandas.concat([temp_prediction,dictionary_cv[key]['results']['test']['predicted_output']],axis=0)
                except: temp_prediction = dictionary_cv[key]['results']['test']['predicted_output']
            temp_prediction = temp_prediction.sort_index()
            del dictionary_cv
        except: temp_prediction = pandas.DataFrame({task:numpy.random.randint(1,size=Y_aal_quan.shape[0])})       
        try: prediction_aal = pandas.concat([prediction_aal,temp_prediction],axis=1)
        except: prediction_aal = temp_prediction
    prediction_aal.to_csv('./Prediction_30/Prediction/Replicas/prediction_aal_rep_'+str(replica)+'.csv',mode='w',index=False)
    try: consensus_prediction_aal = consensus_prediction_aal + prediction_aal
    except: consensus_prediction_aal = prediction_aal
    for task in tasks_brodmann:
        print(task)
        try:
            Y = Y_brodmann_quan[[task]]
            model = models.loc[models['Model'].str.contains('^'+str(task)+'_')]
            dictionary_cv = nn.run_nn_cv(X,Y,k_folds=folds,
                                                    binary_threshold=30,
                                                    imputation_strategy=model['Imputation'].values[0],
                                                    selection_strategy=model['Selection'].values[0],
                                                    target_importance=model['Target importance'].values[0],
                                                    num_epochs=model['Epochs'].values[0],
                                                    scaling_factor=model['Scaling factor'].values[0],
                                                    output_activation=model['Output activation'].values[0],
                                                    regularization=model['Regularization'].values[0],
                                                    dropout_rate=model['Dropout rate'].values[0])
            temp_prediction = pandas.DataFrame()
            for key in dictionary_cv.keys():
                try: temp_prediction = pandas.concat([temp_prediction,dictionary_cv[key]['results']['test']['predicted_output']],axis=0)
                except: temp_prediction = dictionary_cv[key]['results']['test']['predicted_output']
            temp_prediction = temp_prediction.sort_index()
            del dictionary_cv
        except: temp_prediction = pandas.DataFrame({task:numpy.random.randint(1,size=Y_aal_quan.shape[0])})
        try: prediction_brodmann = pandas.concat([prediction_brodmann,temp_prediction],axis=1)
        except: prediction_brodmann = temp_prediction
    prediction_brodmann.to_csv('./Prediction_30/Prediction/Replicas/prediction_brodmann_rep_'+str(replica)+'.csv',mode='w',index=False)
    try: consensus_prediction_brodmann = consensus_prediction_brodmann + prediction_brodmann
    except: consensus_prediction_brodmann = prediction_brodmann
consensus_prediction_aal[consensus_prediction_aal<=replicas/2]=0
consensus_prediction_aal[consensus_prediction_aal>replicas/2]=1
consensus_prediction_brodmann[consensus_prediction_brodmann<=replicas/2]=0
consensus_prediction_brodmann[consensus_prediction_brodmann>replicas/2]=1
consensus_prediction_aal.to_csv('./Prediction_30/Prediction/prediction_aal.csv',mode='w',index=False)
consensus_prediction_brodmann.to_csv('./Prediction_30/Prediction/prediction_brodmann.csv',mode='w',index=False)

#Correct all replicas by applying the absolute rules
#Create a consensus correction considering all replicas

order = ['_'.join([i.split('_')[0],i.split('_')[1]]) for i in list(models['Model'])]
order_aal = [i for i in order if not "brodmann" in i]
order_brodmann = [i for i in order if "brodmann" in i]
relevance_thresholds = [0.5,0.6,0.7,0.8,0.9,0.925,0.950,0.975,1]
for relevance_threshold in relevance_thresholds:
    rules_aal_AD = rd.define_rules_absolute(Y_aal_quan_AD,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_aal)
    rules_aal_FTD = rd.define_rules_absolute(Y_aal_quan_FTD,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_aal)
    rules_aal_CT = rd.define_rules_absolute(Y_aal_quan_CT,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_aal)
    rules_brodmann_AD = rd.define_rules_absolute(Y_brodmann_quan_AD,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_brodmann)
    rules_brodmann_FTD = rd.define_rules_absolute(Y_brodmann_quan_FTD,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_brodmann)
    rules_brodmann_CT = rd.define_rules_absolute(Y_brodmann_quan_CT,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_brodmann)
    for replica in range(replicas):
        print(replica)
        prediction_aal = pandas.read_csv('./Prediction_30/Prediction/Replicas/prediction_aal_rep_'+str(replica)+'.csv')
        prediction_aal_AD = prediction_aal.iloc[Y_aal_quan_AD.index,]
        prediction_aal_FTD = prediction_aal.iloc[Y_aal_quan_FTD.index,]
        prediction_aal_CT = prediction_aal.iloc[Y_aal_quan_CT.index,]
        correction_aal_AD = ra.apply_rules(prediction_aal_AD,rules_aal_AD,'hypometabolic','hypometabolic')
        correction_aal_FTD = ra.apply_rules(prediction_aal_FTD,rules_aal_FTD,'hypometabolic','hypometabolic')
        correction_aal_CT = ra.apply_rules(prediction_aal_CT,rules_aal_CT,'hypometabolic','hypometabolic')
        correction_aal = pandas.concat([correction_aal_AD,correction_aal_FTD,correction_aal_CT],axis=0).sort_index()
        correction_aal.to_csv('./Prediction_30/Absolute/Replicas/correction_aal_'+str(relevance_threshold)+'_rep_'+str(replica)+'.csv',mode='w',index=False)
        try: consensus_correction_aal = consensus_correction_aal + correction_aal
        except: consensus_correction_aal = correction_aal
        prediction_brodmann = pandas.read_csv('./Prediction_30/Prediction/Replicas/prediction_brodmann_rep_'+str(replica)+'.csv')
        prediction_brodmann_AD = prediction_brodmann.iloc[Y_brodmann_quan_AD.index,]
        prediction_brodmann_FTD = prediction_brodmann.iloc[Y_brodmann_quan_FTD.index,]
        prediction_brodmann_CT = prediction_brodmann.iloc[Y_brodmann_quan_CT.index,]
        correction_brodmann_AD = ra.apply_rules(prediction_brodmann_AD,rules_brodmann_AD,'hypometabolic','hypometabolic')
        correction_brodmann_FTD = ra.apply_rules(prediction_brodmann_FTD,rules_brodmann_FTD,'hypometabolic','hypometabolic')
        correction_brodmann_CT = ra.apply_rules(prediction_brodmann_CT,rules_brodmann_CT,'hypometabolic','hypometabolic')
        correction_brodmann = pandas.concat([correction_brodmann_AD,correction_brodmann_FTD,correction_brodmann_CT],axis=0).sort_index()
        correction_brodmann.to_csv('./Prediction_30/Absolute/Replicas/correction_brodmann_'+str(relevance_threshold)+'_rep_'+str(replica)+'.csv',mode='w',index=False)
        try: consensus_correction_brodmann = consensus_correction_brodmann + correction_brodmann
        except: consensus_correction_brodmann = correction_brodmann
    consensus_correction_aal[consensus_correction_aal<=replicas/2]=0
    consensus_correction_aal[consensus_correction_aal>replicas/2]=1
    consensus_correction_brodmann[consensus_correction_brodmann<=replicas/2]=0
    consensus_correction_brodmann[consensus_correction_brodmann>replicas/2]=1
    consensus_correction_aal.to_csv('./Prediction_30/Absolute/correction_aal_'+str(relevance_threshold)+'.csv',mode='w',index=False)
    consensus_correction_brodmann.to_csv('./Prediction_30/Absolute/correction_brodmann_'+str(relevance_threshold)+'.csv',mode='w',index=False)

#Correct all replicas by applying the normalised rules
#Create a consensus correction considering all replicas

order = ['_'.join([i.split('_')[0],i.split('_')[1]]) for i in list(models['Model'])]
order_aal = [i for i in order if not "brodmann" in i]
order_brodmann = [i for i in order if "brodmann" in i]
relevance_thresholds = [0.5,0.6,0.7,0.8,0.9]
for relevance_threshold in relevance_thresholds:
    rules_aal_AD = rd.define_rules_normalised(Y_aal_quan_AD,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_aal)
    rules_aal_FTD = rd.define_rules_normalised(Y_aal_quan_FTD,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_aal)
    rules_aal_CT = rd.define_rules_normalised(Y_aal_quan_CT,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_aal)
    rules_brodmann_AD = rd.define_rules_normalised(Y_brodmann_quan_AD,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_brodmann)
    rules_brodmann_FTD = rd.define_rules_normalised(Y_brodmann_quan_FTD,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_brodmann)
    rules_brodmann_CT = rd.define_rules_normalised(Y_brodmann_quan_CT,30,'non-hierarchical','hypometabolic','hypometabolic',relevance_threshold,False,order_brodmann)
    for replica in range(replicas):
        print(replica)
        prediction_aal = pandas.read_csv('./Prediction_30/Prediction/Replicas/prediction_aal_rep_'+str(replica)+'.csv')
        prediction_aal_AD = prediction_aal.iloc[Y_aal_quan_AD.index,]
        prediction_aal_FTD = prediction_aal.iloc[Y_aal_quan_FTD.index,]
        prediction_aal_CT = prediction_aal.iloc[Y_aal_quan_CT.index,]
        correction_aal_AD = ra.apply_rules(prediction_aal_AD,rules_aal_AD,'hypometabolic','hypometabolic')
        correction_aal_FTD = ra.apply_rules(prediction_aal_FTD,rules_aal_FTD,'hypometabolic','hypometabolic')
        correction_aal_CT = ra.apply_rules(prediction_aal_CT,rules_aal_CT,'hypometabolic','hypometabolic')
        correction_aal = pandas.concat([correction_aal_AD,correction_aal_FTD,correction_aal_CT],axis=0).sort_index()
        correction_aal.to_csv('./Prediction_30/Normalised/Replicas/correction_aal_'+str(relevance_threshold)+'_rep_'+str(replica)+'.csv',mode='w',index=False)
        try: consensus_correction_aal = consensus_correction_aal + correction_aal
        except: consensus_correction_aal = correction_aal
        prediction_brodmann = pandas.read_csv('./Prediction_30/Prediction/Replicas/prediction_brodmann_rep_'+str(replica)+'.csv')
        prediction_brodmann_AD = prediction_brodmann.iloc[Y_brodmann_quan_AD.index,]
        prediction_brodmann_FTD = prediction_brodmann.iloc[Y_brodmann_quan_FTD.index,]
        prediction_brodmann_CT = prediction_brodmann.iloc[Y_brodmann_quan_CT.index,]
        correction_brodmann_AD = ra.apply_rules(prediction_brodmann_AD,rules_brodmann_AD,'hypometabolic','hypometabolic')
        correction_brodmann_FTD = ra.apply_rules(prediction_brodmann_FTD,rules_brodmann_FTD,'hypometabolic','hypometabolic')
        correction_brodmann_CT = ra.apply_rules(prediction_brodmann_CT,rules_brodmann_CT,'hypometabolic','hypometabolic')
        correction_brodmann = pandas.concat([correction_brodmann_AD,correction_brodmann_FTD,correction_brodmann_CT],axis=0).sort_index()
        correction_brodmann.to_csv('./Prediction_30/Normalised/Replicas/correction_brodmann_'+str(relevance_threshold)+'_rep_'+str(replica)+'.csv',mode='w',index=False)
        try: consensus_correction_brodmann = consensus_correction_brodmann + correction_brodmann
        except: consensus_correction_brodmann = correction_brodmann
    consensus_correction_aal[consensus_correction_aal<=replicas/2]=0
    consensus_correction_aal[consensus_correction_aal>replicas/2]=1
    consensus_correction_brodmann[consensus_correction_brodmann<=replicas/2]=0
    consensus_correction_brodmann[consensus_correction_brodmann>replicas/2]=1
    consensus_correction_aal.to_csv('./Prediction_30/Normalised/correction_aal_'+str(relevance_threshold)+'.csv',mode='w',index=False)
    consensus_correction_brodmann.to_csv('./Prediction_30/Normalised/correction_brodmann_'+str(relevance_threshold)+'.csv',mode='w',index=False)
