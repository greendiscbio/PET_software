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
GEN = 300
FIL = 0.5
POS = 1

#Define the parameters that will vary

SVM1 = MonoObjectiveCV(estimator=SVC(C=2,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
SVM2 = MonoObjectiveCV(estimator=SVC(C=10,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
SVM3 = MonoObjectiveCV(estimator=SVC(C=50,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
AN1 = 0.2
AN2 = 0.3
AN3 = 0.4
EL1 = 0.2
EL2 = 0.3
EL3 = 0.4
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
        for k in range(0,20):
            SVM = random.choices(['SVM1','SVM2','SVM3'],[0,1,0],k=1)[0]
            AN = random.choices(['AN1','AN2','AN3'],[0,2/3,1/3],k=1)[0]
            EL = random.choices(['EL1','EL2','EL3'],[1/4,1/4,2/4],k=1)[0]
            MR = random.choices(['MR1','MR2','MR3'],[2/4,1/4,1/4],k=1)[0]
            SE = random.choices(['SE1','SE2','SE3'],[1/3,1/3,1/3],k=1)[0]
            combination = [SVM,AN,EL,MR,SE]
            ga = BasicGA(population_size=POP,generations=GEN,fitness=globals()[combination[0]],annihilation=globals()[combination[1]],fill_with_elite=FIL,elitism=globals()[combination[2]],mutation_rate=globals()[combination[3]],selection=globals()[combination[4]],positive_class=POS,id='_'.join(combination)+'_'+str(task))
            par.append(ga)
            del ga
        try:
            par_ga = Parallel(*par)
            par_ga.set_features(features)
            par_ga.fit(x_data,y_data)
            par_ga.save(dir_name='./Optimization_mono_30/'+str(task),overwrite=True)
            del par, par_ga
        except: pass
for task in tasks_brodmann:
    y_data = Y_brodmann[[task]].to_numpy().flatten()
    if numpy.sum(y_data)!=0:
        par = []
        for k in range(0,20):
            SVM = random.choices(['SVM1','SVM2','SVM3'],[0,1,0],k=1)[0]
            AN = random.choices(['AN1','AN2','AN3'],[0,2/3,1/3],k=1)[0]
            EL = random.choices(['EL1','EL2','EL3'],[1/4,1/4,2/4],k=1)[0]
            MR = random.choices(['MR1','MR2','MR3'],[2/4,1/4,1/4],k=1)[0]
            SE = random.choices(['SE1','SE2','SE3'],[1/3,1/3,1/3],k=1)[0]
            combination = [SVM,AN,EL,MR,SE]
            ga = BasicGA(population_size=POP,generations=GEN,fitness=globals()[combination[0]],annihilation=globals()[combination[1]],fill_with_elite=FIL,elitism=globals()[combination[2]],mutation_rate=globals()[combination[3]],selection=globals()[combination[4]],positive_class=POS,id='_'.join(combination)+'_'+str(task))
            par.append(ga)
            del ga
        try:
            par_ga = Parallel(*par)
            par_ga.set_features(features)
            par_ga.fit(x_data,y_data)
            par_ga.save(dir_name='./Optimization_mono_30/'+str(task),overwrite=True)
            del par, par_ga
        except: pass

#Register the results for each region

f1 = open('./Optimization_mono_30/optimization_train.csv',mode='w')
f2 = open('./Optimization_mono_30/optimization_test.csv',mode='w')
f3 = open('./Optimization_mono_30/optimization_features.csv',mode='w')
f1.write('accuracy_mean,precision_mean,recall_mean,f1_mean,accuracy_std,precision_std,recall_std,f1_std,Model\n')
f2.write('accuracy_mean,precision_mean,recall_mean,f1_mean,accuracy_std,precision_std,recall_std,f1_std,Model\n')
f3.write('Features,Model\n')
f1.close()
f2.close()
f3.close()
f1 = open('./Optimization_mono_30/optimization_train.csv',mode='a')
f2 = open('./Optimization_mono_30/optimization_test.csv',mode='a')
f3 = open('./Optimization_mono_30/optimization_features.csv',mode='a')
for task in tasks_aal+tasks_brodmann:
    try:
        print(task)
        names = [name for name in os.listdir("./Optimization_mono_30/"+str(task))]
        models = [BasicGA.load(model,"./Optimization_mono_30/"+str(task)) for model in names]
        for name,model in zip(names,models):
            try:
                train,test = Evaluator(model).metrics_table(cv=5,reps=5); train['Model'] = name; test['Model'] = name
                features = pandas.DataFrame({'features':[model.best_features],'Model':[name]})
                train.to_csv('./Optimization_mono_30/optimization_train.csv',mode='a',index=False,header=False)
                test.to_csv('./Optimization_mono_30/optimization_test.csv',mode='a',index=False,header=False)
                features.to_csv('./Optimization_mono_30/optimization_features.csv',mode='a',index=False,header=False)
            except: pass
    except: pass
f1.close()
f2.close()
f3.close()

# Save the best models for each region

train = pandas.read_csv('./Optimization_mono_30/optimization_train.csv')
test = pandas.read_csv('./Optimization_mono_30/optimization_test.csv')
features = pandas.read_csv('./Optimization_mono_30/optimization_features.csv')
regions = tasks_aal+tasks_brodmann
criteria = ['accuracy','f1']
mo.keep_best(train,test,features,regions,criteria,'./Optimization_mono_30/optimization')