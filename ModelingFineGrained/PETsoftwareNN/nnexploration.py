#Import all packages and module dependencies

import itertools
import sys
sys.path.insert(0,'../../Tools')
from nntools import *
nn = NeuralNetwork()

#Load the data

X_20 = pandas.read_csv('../../Data/X_20.csv')
X_25 = pandas.read_csv('../../Data/X_25.csv')
X_30 = pandas.read_csv('../../Data/X_30.csv')
Y_aal = pandas.read_csv('../../Data/Y_aal_quan.csv')
Y_brodmann = pandas.read_csv('../../Data/Y_brodmann_quan.csv')
tasks_aal = ['f1_l','t1_l','sma_r','o2_r','p2_r','pcin_r','put_l','pcin_l','p2_l','mcin_r','f2o_r','f1mo_r','ag_r','acin_r','ling_r','in_l','hip_r','amyg_r']
tasks_brodmann = ['b47','b37','b16','b35','b33','b20']
replicas = 5

#Explore the parameters

#IMPUTATION

imputation_strategy = ['mean','median','knn','bayes']
exploration = numpy.repeat(imputation_strategy,replicas)

f1 = open('./Exploration/imputation_train.csv',mode='w')
f2 = open('./Exploration/imputation_test.csv',mode='w')
f3 = open('./Exploration/imputation_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Imputation,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/imputation_train.csv',mode='a')
f2 = open('./Exploration/imputation_test.csv',mode='a')
f3 = open('./Exploration/imputation_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,imputation_strategy=exp)
            dictionary['parameters'] = dictionary['parameters'][['Imputation']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/imputation')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,imputation_strategy=exp)
            dictionary['parameters'] = dictionary['parameters'][['Imputation']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/imputation')
        except: pass 
f1.close()
f2.close()
f3.close()

#FEATURE SELECTION

target_importance = [0.5,0.6,0.7,0.8,0.9,1]
selection_strategy = ['pca','chi','anova']
exploration = numpy.array(list(itertools.product(target_importance,selection_strategy)))
exploration = numpy.repeat(exploration,replicas,axis=0)

f1 = open('./Exploration/feature_selection_train.csv',mode='w')
f2 = open('./Exploration/feature_selection_test.csv',mode='w')
f3 = open('./Exploration/feature_selection_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Selection,Target importance,Features,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/feature_selection_train.csv',mode='a')
f2 = open('./Exploration/feature_selection_test.csv',mode='a')
f3 = open('./Exploration/feature_selection_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,target_importance=float(exp[0]),selection_strategy=exp[1])
            dictionary['parameters'] = dictionary['parameters'][['Selection','Target importance','Features']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/feature_selection')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,target_importance=float(exp[0]),selection_strategy=exp[1])
            dictionary['parameters'] = dictionary['parameters'][['Selection','Target importance','Features']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/feature_selection')
        except: pass
f1.close()
f2.close()
f3.close()

#SCALING FACTOR

scaling_factor = [1,1.2,1.4,1.6,1.8,2]
exploration = numpy.repeat(scaling_factor,replicas)

f1 = open('./Exploration/scaling_factor_train.csv',mode='w')
f2 = open('./Exploration/scaling_factor_test.csv',mode='w')
f3 = open('./Exploration/scaling_factor_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Scaling factor,Hidden units,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/scaling_factor_train.csv',mode='a')
f2 = open('./Exploration/scaling_factor_test.csv',mode='a')
f3 = open('./Exploration/scaling_factor_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,scaling_factor=float(exp))
            dictionary['parameters'] = dictionary['parameters'][['Scaling factor','Hidden units']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/scaling_factor')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,scaling_factor=float(exp))
            dictionary['parameters'] = dictionary['parameters'][['Scaling factor','Hidden units']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/scaling_factor')
        except: pass
f1.close()
f2.close()
f3.close()

#OUTPUT ACTIVATION

output_activation = ['sigmoid','softmax']
exploration = numpy.repeat(output_activation,replicas)

f1 = open('./Exploration/output_activation_train.csv',mode='w')
f2 = open('./Exploration/output_activation_test.csv',mode='w')
f3 = open('./Exploration/output_activation_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Output activation,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/output_activation_train.csv',mode='a')
f2 = open('./Exploration/output_activation_test.csv',mode='a')
f3 = open('./Exploration/output_activation_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,output_activation=exp)
            dictionary['parameters'] = dictionary['parameters'][['Output activation']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/output_activation')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,output_activation=exp)
            dictionary['parameters'] = dictionary['parameters'][['Output activation']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/output_activation')
        except: pass
f1.close()
f2.close()
f3.close()

#BALANCING

portion_negative = [0.5,0.6,0.7,0.8,0.9]
exploration = numpy.repeat(portion_negative,replicas)

f1 = open('./Exploration/balancing_train.csv',mode='w')
f2 = open('./Exploration/balancing_test.csv',mode='w')
f3 = open('./Exploration/balancing_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Balancing,Negatives,Patients,Weighting,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/balancing_train.csv',mode='a')
f2 = open('./Exploration/balancing_test.csv',mode='a')
f3 = open('./Exploration/balancing_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,balancing=True,portion_negative=float(exp),minimum_size=0.2,weighting=False)
            dictionary['parameters'] = dictionary['parameters'][['Balancing','Negatives','Patients','Weighting']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/balancing')
        except: pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,balancing=True,portion_negative=float(exp),minimum_size=0.2,weighting=False)
            dictionary['parameters'] = dictionary['parameters'][['Balancing','Negatives','Patients','Weighting']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/balancing')
        except: pass
f1.close()
f2.close()
f3.close()

#WEIGHTING

class_weights = ['Default',{0:0.5,1:0.5},{0:0.4,1:0.6},{0:0.3,1:0.7},{0:0.2,1:0.8},{0:0.1,1:0.9}]
exploration = numpy.repeat(class_weights,replicas)

f1 = open('./Exploration/weighting_train.csv',mode='w')
f2 = open('./Exploration/weighting_test.csv',mode='w')
f3 = open('./Exploration/weighting_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Balancing,Weighting,Class weights,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/weighting_train.csv',mode='a')
f2 = open('./Exploration/weighting_test.csv',mode='a')
f3 = open('./Exploration/weighting_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,balancing=False,weighting=True,class_weights=exp)
            dictionary['parameters'] = dictionary['parameters'][['Balancing','Weighting','Class weights']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/weighting')
        except: pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,balancing=False,weighting=True,class_weights=exp)
            dictionary['parameters'] = dictionary['parameters'][['Balancing','Weighting','Class weights']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/weighting')
        except: pass
f1.close()
f2.close()
f3.close()

#EPOCHS

num_epochs = [100,150,200,250,300]
exploration = numpy.repeat(num_epochs,replicas)

f1 = open('./Exploration/epochs_train.csv',mode='w')
f2 = open('./Exploration/epochs_test.csv',mode='w')
f3 = open('./Exploration/epochs_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Epochs,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/epochs_train.csv',mode='a')
f2 = open('./Exploration/epochs_test.csv',mode='a')
f3 = open('./Exploration/epochs_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=exp)
            dictionary['parameters'] = dictionary['parameters'][['Epochs']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/epochs')
        except: pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=exp)
            dictionary['parameters'] = dictionary['parameters'][['Epochs']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/epochs')
        except: pass
f1.close()
f2.close()
f3.close()

#DROPOUT

num_epochs = [200,250,300]
dropout_rate = [0.2,0.3,0.4]
exploration = numpy.array(list(itertools.product(num_epochs,dropout_rate)))
exploration = numpy.repeat(exploration,replicas,axis=0)

f1 = open('./Exploration/dropout_train.csv',mode='w')
f2 = open('./Exploration/dropout_test.csv',mode='w')
f3 = open('./Exploration/dropout_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Epochs,Regularization,Dropout rate,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/dropout_train.csv',mode='a')
f2 = open('./Exploration/dropout_test.csv',mode='a')
f3 = open('./Exploration/dropout_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=int(exp[0]),regularization='dropout',dropout_rate=float(exp[1]))
            dictionary['parameters'] = dictionary['parameters'][['Epochs','Regularization','Dropout rate']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/dropout')
        except:pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=int(exp[0]),regularization='dropout',dropout_rate=float(exp[1]))
            dictionary['parameters'] = dictionary['parameters'][['Epochs','Regularization','Dropout rate']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/dropout')
        except: pass
f1.close()
f2.close()
f3.close()

#L2 REGULARIZATION

num_epochs = [200,250,300]
l2_rate = [0.01,0.001,0.0001]
exploration = numpy.array(list(itertools.product(num_epochs,l2_rate)))
exploration = numpy.repeat(exploration,replicas,axis=0)

f1 = open('./Exploration/l2_train.csv',mode='w')
f2 = open('./Exploration/l2_test.csv',mode='w')
f3 = open('./Exploration/l2_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Epochs,Regularization,L2 rate,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/l2_train.csv',mode='a')
f2 = open('./Exploration/l2_test.csv',mode='a')
f3 = open('./Exploration/l2_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=int(exp[0]),regularization='l2',l2_rate=float(exp[1]))
            dictionary['parameters'] = dictionary['parameters'][['Epochs','Regularization','L2 rate']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/l2')
        except: pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=int(exp[0]),regularization='l2',l2_rate=float(exp[1]))
            dictionary['parameters'] = dictionary['parameters'][['Epochs','Regularization','L2 rate']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/l2')
        except: pass
f1.close()
f2.close()
f3.close()

#IMPUTATION (VARIATION 1)

imputation_strategy = ['mean','median','knn','bayes']
exploration = numpy.repeat(imputation_strategy,replicas)

f1 = open('./Exploration/imputation_var1_train.csv',mode='w')
f2 = open('./Exploration/imputation_var1_test.csv',mode='w')
f3 = open('./Exploration/imputation_var1_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Imputation,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/imputation_var1_train.csv',mode='a')
f2 = open('./Exploration/imputation_var1_test.csv',mode='a')
f3 = open('./Exploration/imputation_var1_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_25,Y,imputation_strategy=exp)
            dictionary['parameters'] = dictionary['parameters'][['Imputation']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/imputation_var1')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_25,Y,imputation_strategy=exp)
            dictionary['parameters'] = dictionary['parameters'][['Imputation']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/imputation_var1')
        except: pass 
f1.close()
f2.close()
f3.close()

#IMPUTATION (VARIATION 2)

imputation_strategy = ['mean','median','knn','bayes']
exploration = numpy.repeat(imputation_strategy,replicas)

f1 = open('./Exploration/imputation_var2_train.csv',mode='w')
f2 = open('./Exploration/imputation_var2_test.csv',mode='w')
f3 = open('./Exploration/imputation_var2_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Imputation,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/imputation_var2_train.csv',mode='a')
f2 = open('./Exploration/imputation_var2_test.csv',mode='a')
f3 = open('./Exploration/imputation_var2_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_30,Y,imputation_strategy=exp)
            dictionary['parameters'] = dictionary['parameters'][['Imputation']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/imputation_var2')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_30,Y,imputation_strategy=exp)
            dictionary['parameters'] = dictionary['parameters'][['Imputation']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/imputation_var2')
        except: pass 
f1.close()
f2.close()
f3.close()

#FEATURE SELECTION (VARIATION 1)

target_importance = [0.5,0.6,0.7,0.8,0.9,1]
selection_strategy = ['pca','chi','anova']
exploration = numpy.array(list(itertools.product(target_importance,selection_strategy)))
exploration = numpy.repeat(exploration,replicas,axis=0)

f1 = open('./Exploration/feature_selection_var1_train.csv',mode='w')
f2 = open('./Exploration/feature_selection_var1_test.csv',mode='w')
f3 = open('./Exploration/feature_selection_var1_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Selection,Target importance,Features,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/feature_selection_var1_train.csv',mode='a')
f2 = open('./Exploration/feature_selection_var1_test.csv',mode='a')
f3 = open('./Exploration/feature_selection_var1_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_25,Y,target_importance=float(exp[0]),selection_strategy=exp[1],imputation_strategy='knn')
            dictionary['parameters'] = dictionary['parameters'][['Selection','Target importance','Features']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/feature_selection_var1')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_25,Y,target_importance=float(exp[0]),selection_strategy=exp[1],imputation_strategy='knn')
            dictionary['parameters'] = dictionary['parameters'][['Selection','Target importance','Features']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/feature_selection_var1')
        except: pass
f1.close()
f2.close()
f3.close()

#FEATURE SELECTION (VARIATION 2)

target_importance = [0.5,0.6,0.7,0.8,0.9,1]
selection_strategy = ['pca','chi','anova']
exploration = numpy.array(list(itertools.product(target_importance,selection_strategy)))
exploration = numpy.repeat(exploration,replicas,axis=0)

f1 = open('./Exploration/feature_selection_var2_train.csv',mode='w')
f2 = open('./Exploration/feature_selection_var2_test.csv',mode='w')
f3 = open('./Exploration/feature_selection_var2_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Selection,Target importance,Features,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/feature_selection_var2_train.csv',mode='a')
f2 = open('./Exploration/feature_selection_var2_test.csv',mode='a')
f3 = open('./Exploration/feature_selection_var2_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_30,Y,target_importance=float(exp[0]),selection_strategy=exp[1],imputation_strategy='knn')
            dictionary['parameters'] = dictionary['parameters'][['Selection','Target importance','Features']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/feature_selection_var2')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_30,Y,target_importance=float(exp[0]),selection_strategy=exp[1],imputation_strategy='knn')
            dictionary['parameters'] = dictionary['parameters'][['Selection','Target importance','Features']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/feature_selection_var2')
        except: pass
f1.close()
f2.close()
f3.close()

#FEATURE SELECTION (VARIATION 3)

target_importance = [0.5,0.6,0.7,0.8,0.9,1]
selection_strategy = ['pca','chi','anova']
exploration = numpy.array(list(itertools.product(target_importance,selection_strategy)))
exploration = numpy.repeat(exploration,replicas,axis=0)

f1 = open('./Exploration/feature_selection_var3_train.csv',mode='w')
f2 = open('./Exploration/feature_selection_var3_test.csv',mode='w')
f3 = open('./Exploration/feature_selection_var3_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Selection,Target importance,Features,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/feature_selection_var3_train.csv',mode='a')
f2 = open('./Exploration/feature_selection_var3_test.csv',mode='a')
f3 = open('./Exploration/feature_selection_var3_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,target_importance=float(exp[0]),selection_strategy=exp[1])
            dictionary['parameters'] = dictionary['parameters'][['Selection','Target importance','Features']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/feature_selection_var3')
        except: pass            
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,target_importance=float(exp[0]),selection_strategy=exp[1])
            dictionary['parameters'] = dictionary['parameters'][['Selection','Target importance','Features']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/feature_selection_var3')
        except: pass
f1.close()
f2.close()
f3.close()

#BALANCING (VARIATION 1)

portion_negative = [0.5,0.6,0.7,0.8,0.9]
exploration = numpy.repeat(portion_negative,replicas)

f1 = open('./Exploration/balancing_var1_train.csv',mode='w')
f2 = open('./Exploration/balancing_var1_test.csv',mode='w')
f3 = open('./Exploration/balancing_var1_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Balancing,Negatives,Patients,Weighting,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/balancing_var1_train.csv',mode='a')
f2 = open('./Exploration/balancing_var1_test.csv',mode='a')
f3 = open('./Exploration/balancing_var1_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,balancing=True,portion_negative=float(exp),minimum_size=0.2,weighting=True)
            dictionary['parameters'] = dictionary['parameters'][['Balancing','Negatives','Patients','Weighting']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/balancing_var1')
        except: pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,balancing=True,portion_negative=float(exp),minimum_size=0.2,weighting=True)
            dictionary['parameters'] = dictionary['parameters'][['Balancing','Negatives','Patients','Weighting']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/balancing_var1')
        except: pass
f1.close()
f2.close()
f3.close()

#EPOCHS (VARIATION 1)

num_epochs = [100,150,200,250,300]
exploration = numpy.repeat(num_epochs,replicas)

f1 = open('./Exploration/epochs_var1_train.csv',mode='w')
f2 = open('./Exploration/epochs_var1_test.csv',mode='w')
f3 = open('./Exploration/epochs_var1_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Epochs,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/epochs_var1_train.csv',mode='a')
f2 = open('./Exploration/epochs_var1_test.csv',mode='a')
f3 = open('./Exploration/epochs_var1_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=exp,balancing=True,portion_negative=0.5,minimum_size=0.2,weighting=False)
            dictionary['parameters'] = dictionary['parameters'][['Epochs']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/epochs_var1')
        except: pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=exp,balancing=True,portion_negative=0.5,minimum_size=0.2,weighting=False)
            dictionary['parameters'] = dictionary['parameters'][['Epochs']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/epochs_var1')
        except: pass
f1.close()
f2.close()
f3.close()

#EPOCHS (VARIATION 2)

num_epochs = [100,150,200,250,300]
exploration = numpy.repeat(num_epochs,replicas)

f1 = open('./Exploration/epochs_var2_train.csv',mode='w')
f2 = open('./Exploration/epochs_var2_test.csv',mode='w')
f3 = open('./Exploration/epochs_var2_parameters.csv',mode='w')
f1.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f2.write('TP,TN,FP,FN,accuracy,precision,recall,f1,Model\n')
f3.write('Epochs,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Exploration/epochs_var2_train.csv',mode='a')
f2 = open('./Exploration/epochs_var2_test.csv',mode='a')
f3 = open('./Exploration/epochs_var2_parameters.csv',mode='a')
for task in tasks_aal:
    print(task)
    Y = Y_aal[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=exp,weighting=False)
            dictionary['parameters'] = dictionary['parameters'][['Epochs']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/epochs_var2')
        except: pass
for task in tasks_brodmann:
    print(task)
    Y = Y_brodmann[[task]]
    k=0
    for exp in exploration:
        k=k+1
        try:
            dictionary = nn.run_nn(X_20,Y,num_epochs=exp,weighting=False)
            dictionary['parameters'] = dictionary['parameters'][['Epochs']]
            nn.record_nn(dictionary,str(task)+'_'+str(k),'./Exploration/epochs_var2')
        except: pass
f1.close()
f2.close()
f3.close()