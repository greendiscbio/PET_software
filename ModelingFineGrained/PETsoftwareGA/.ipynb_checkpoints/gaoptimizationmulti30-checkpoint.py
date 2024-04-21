#Import all packages and module dependencies

import sys
sys.path.insert(0,'../../Tools')
from gatools import *
bi = Binarization()
im = Imputation()
no = Normalization()
tr = Transformations()
mo = Models()

#Load the data

X = pandas.read_csv('../../Data/X_25_diag.csv')
X_imputed = im.imputate(X,'bayes')
X_first_encoded = tr.encode_column(X_imputed,'diagnostic',{3:'FTD',5:'AD',7:'AD',13:'CT',16:'CT',17:'AD'})
X_second_encoded = tr.encode_column(X_first_encoded,'sex',{1:'male',2:'female'})
X_normalised = no.normalise(X_second_encoded,'minmax')
x_data = X_normalised.to_numpy()
features = X_normalised.columns.to_numpy()
Y_aal = bi.get_binary(pandas.read_csv('../../Data/Y_aal_quan.csv'),30)
Y_brodmann = bi.get_binary(pandas.read_csv('../../Data/Y_brodmann_quan.csv'),30)
tasks_aal = list(Y_aal.columns)
tasks_brodmann = list(Y_brodmann.columns)

#Define the parameters that will not vary

POP = 50
GEN = 600
POS = 1

#Define the parameters that will vary

ALG1 = 'NSGA2'
ALG2 = 'SPEA2'
SVM1 = MonoObjectiveCV(estimator=SVC(C=2,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
SVM2 = MonoObjectiveCV(estimator=SVC(C=10,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
SVM3 = MonoObjectiveCV(estimator=SVC(C=50,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
MR1 = 0.05
MR2 = 0.1
MR3 = 0.15
SE1 = TournamentSelection(k=2,replacement=False,winners=1)
SE2 = TournamentSelection(k=5,replacement=False,winners=2)
SE3 = RouletteWheel()

#Define and run the genetic algorithms

for task in tasks_aal:
    y_data = Y_aal[[task]].to_numpy().flatten()
    if numpy.sum(y_data)!=0:
        par = []
        for k in range(0,10):
            ALG = random.choices(['ALG1','ALG2'],[1/2,1/2],k=1)[0]
            SVM = random.choices(['SVM1','SVM2','SVM3'],[3/10,6/10,1/10],k=1)[0]
            MR = random.choices(['MR1','MR2','MR3'],[2/5,2/5,1/5],k=1)[0]
            SE = random.choices(['SE1','SE2','SE3'],[5/10,3/10,2/10],k=1)[0]
            combination = [ALG,SVM,MR,SE]
            if globals()[combination[0]]=='NSGA2':ga = NSGA2(population_size=POP,generations=GEN,fitness=globals()[combination[1]],optimize_features=True,mutation_rate=globals()[combination[2]],selection=globals()[combination[3]],positive_class=POS,id='_'.join(combination)+'_'+str(task))
            elif globals()[combination[0]]=='SPEA2': ga = SPEA2(population_size=POP,generations=GEN,fitness=globals()[combination[1]],optimize_features=True,mutation_rate=globals()[combination[2]],selection=globals()[combination[3]],positive_class=POS,id='_'.join(combination)+'_'+str(task))
            par.append(ga)
            del ga
        try:
            par_ga = Parallel(*par)
            par_ga.set_features(features)
            par_ga.fit(x_data,y_data)
            par_ga.save(dir_name='./Optimization_multi_30/'+str(task),overwrite=True)
            del par, par_ga
        except: pass
for task in tasks_brodmann:
    y_data = Y_brodmann[[task]].to_numpy().flatten()
    if numpy.sum(y_data)!=0:
        par = []
        for k in range(0,10):
            ALG = random.choices(['ALG1','ALG2'],[5/10,5/10],k=1)[0]
            SVM = random.choices(['SVM1','SVM2','SVM3'],[3/10,6/10,1/10],k=1)[0]
            MR = random.choices(['MR1','MR2','MR3'],[4/10,4/10,2/10],k=1)[0]
            SE = random.choices(['SE1','SE2','SE3'],[5/10,3/10,2/10],k=1)[0]
            combination = [ALG,SVM,MR,SE]
            if globals()[combination[0]]=='NSGA2':ga = NSGA2(population_size=POP,generations=GEN,fitness=globals()[combination[1]],optimize_features=True,mutation_rate=globals()[combination[2]],selection=globals()[combination[3]],positive_class=POS,id='_'.join(combination)+'_'+str(task))
            elif globals()[combination[0]]=='SPEA2': ga = SPEA2(population_size=POP,generations=GEN,fitness=globals()[combination[1]],optimize_features=True,mutation_rate=globals()[combination[2]],selection=globals()[combination[3]],positive_class=POS,id='_'.join(combination)+'_'+str(task))
            par.append(ga)
            del ga
        try:
            par_ga = Parallel(*par)
            par_ga.set_features(features)
            par_ga.fit(x_data,y_data)
            par_ga.save(dir_name='./Optimization_multi_30/'+str(task),overwrite=True)
            del par, par_ga
        except: pass

#Register the results for each region

f1 = open('./Optimization_multi_30/optimization_train.csv',mode='w')
f2 = open('./Optimization_multi_30/optimization_test.csv',mode='w')
f3 = open('./Optimization_multi_30/optimization_features.csv',mode='w')
f1.write('accuracy_mean,precision_mean,recall_mean,f1_mean,accuracy_std,precision_std,recall_std,f1_std,Model\n')
f2.write('accuracy_mean,precision_mean,recall_mean,f1_mean,accuracy_std,precision_std,recall_std,f1_std,Model\n')
f3.write('Features,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Optimization_multi_30/optimization_train.csv',mode='a')
f2 = open('./Optimization_multi_30/optimization_test.csv',mode='a')
f3 = open('./Optimization_multi_30/optimization_features.csv',mode='a')
for task in tasks_aal+tasks_brodmann:
    try:
        print(task)
        names = [name for name in os.listdir("./Optimization_multi_30/"+str(task))]
        models = [BasicGA.load(model,"./Optimization_multi_30/"+str(task)) for model in names]
        for name,model in zip(names,models):
            for index in range(len(model.best_features)):
                try:
                    train,test = MultiObjectiveEvaluator(model).metrics_table(idx=index,cv=5,reps=5)
                    train['Model'] = str(index)+'_'+name; test['Model'] = str(index)+'_'+name
                    features = pandas.DataFrame({'features':[model.best_features[index]],'Model':[str(index)+'_'+name]})
                    train.to_csv('./Optimization_multi_30/optimization_train.csv',mode='a',index=False,header=False)
                    test.to_csv('./Optimization_multi_30/optimization_test.csv',mode='a',index=False,header=False)
                    features.to_csv('./Optimization_multi_30/optimization_features.csv',mode='a',index=False,header=False)
                except: pass
    except: pass
f1.close()
f2.close()
f3.close()

# Save the best models for each region

train = pandas.read_csv('./Optimization_multi_30/optimization_train.csv')
test = pandas.read_csv('./Optimization_multi_30/optimization_test.csv')
features = pandas.read_csv('./Optimization_multi_30/optimization_features.csv')
regions = tasks_aal+tasks_brodmann
criteria = ['accuracy','f1']
mo.keep_best(train,test,features,regions,criteria,'./Optimization_multi_30/optimization')