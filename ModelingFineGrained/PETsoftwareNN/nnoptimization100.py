#Import all packages and module dependencies

import sys
sys.path.insert(0,'../../Tools')
from nntools import *
nn = NeuralNetwork()
tr = Transformations()

#Load the data

X = pandas.read_csv('../../Data/X_25_diag.csv')
X_first_encoded = tr.encode_column(X,'diagnostic',{3:'FTD',5:'AD',7:'AD',13:'CT',16:'CT',17:'AD'})
X_second_encoded = tr.encode_column(X_first_encoded,'sex',{1:'male',2:'female'})
X = X_second_encoded
Y_aal = pandas.read_csv('../../Data/Y_aal_quan.csv')
Y_brodmann = pandas.read_csv('../../Data/Y_brodmann_quan.csv')
replicas = 20
folds = 5
tasks_aal = list(Y_aal.columns)
tasks_brodmann = list(Y_brodmann.columns)

#Define and run the neural networks
#Register the results for each region

f1 = open('./Optimization_100/optimization_train.csv',mode='w')
f2 = open('./Optimization_100/optimization_test.csv',mode='w')
f3 = open('./Optimization_100/optimization_parameters.csv',mode='w')
f1.write('TP_mean,TN_mean,FP_mean,FN_mean,accuracy_mean,precision_mean,recall_mean,f1_mean,TP_std,TN_std,FP_std,FN_std,accuracy_std,precision_std,recall_std,f1_std,Model\n')
f2.write('TP_mean,TN_mean,FP_mean,FN_mean,accuracy_mean,precision_mean,recall_mean,f1_mean,TP_std,TN_std,FP_std,FN_std,accuracy_std,precision_std,recall_std,f1_std,Model\n')
f3.write('Imputation,Selection,Target importance,Features,Balancing,Negatives,Minimum size,Patients,Input normalisation,Epochs,Scaling factor,Hidden units,Output activation,Weighting,Class weights,Regularization,Dropout rate,L2 rate,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Optimization_100/optimization_train.csv',mode='a')
f2 = open('./Optimization_100/optimization_test.csv',mode='a')
f3 = open('./Optimization_100/optimization_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    for k in range(0,replicas):
        try:
            dictionary_cv = nn.run_nn_cv(X,Y,k_folds=folds,
                                                    binary_threshold=100,
                                                    imputation_strategy=random.choice(['knn','bayes']),
                                                    selection_strategy=random.choice(['anova','chi']),
                                                    target_importance=1-float(numpy.random.exponential(0.05,1)),
                                                    num_epochs=int(numpy.random.normal(250,25,1)),
                                                    scaling_factor=float(numpy.random.normal(1.5,0.1,1)),
                                                    output_activation=random.choice(['sigmoid','softmax']),
                                                    regularization='dropout',
                                                    dropout_rate=float(numpy.random.normal(0.4,0.025,1)))
            nn.record_nn_cv(dictionary_cv,folds,str(task)+'_'+str(k),'./Optimization_100/optimization')
            del dictionary_cv
        except:pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    for k in range(0,replicas):
        try:
            dictionary_cv = nn.run_nn_cv(X,Y,k_folds=folds,
                                                    binary_threshold=100,
                                                    imputation_strategy=random.choice(['knn','bayes']),
                                                    selection_strategy=random.choice(['anova','chi']),
                                                    target_importance=1-float(numpy.random.exponential(0.05,1)),
                                                    num_epochs=int(numpy.random.normal(250,25,1)),
                                                    scaling_factor=float(numpy.random.normal(1.5,0.1,1)),
                                                    output_activation=random.choice(['sigmoid','softmax']),
                                                    regularization='dropout',
                                                    dropout_rate=float(numpy.random.normal(0.4,0.025,1)))
            nn.record_nn_cv(dictionary_cv,folds,str(task)+'_'+str(k),'./Optimization_100/optimization')
            del dictionary_cv
        except: pass 
f1.close()
f2.close()
f3.close()

# Register the best models for each region

train = pandas.read_csv('./Optimization_100/optimization_train.csv')
test = pandas.read_csv('./Optimization_100/optimization_test.csv')
parameters = pandas.read_csv('./Optimization_100/optimization_parameters.csv')
regions = tasks_aal+tasks_brodmann
criteria = ['accuracy','f1']
nn.keep_best(train,test,parameters,regions,criteria,'./Optimization_100/optimization')