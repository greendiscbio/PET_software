#Import all packages and module dependencies

import sys
sys.path.insert(0,'../../Tools')
from gatools import *
bi = Binarization()
im = Imputation()
no = Normalization()
tr = Transformations()

#Load the data

X = pandas.read_csv('../../Data/X_25_diag.csv')
X_imputed = im.imputate(X,'bayes')
X_first_encoded = tr.encode_column(X_imputed,'diagnostic',{3:'FTD',5:'AD',7:'AD',13:'CT',16:'CT',17:'AD'})
X_second_encoded = tr.encode_column(X_first_encoded,'sex',{1:'male',2:'female'})
X_normalised = no.normalise(X_second_encoded,'minmax')
x_data = X_normalised.to_numpy()
features = X_normalised.columns.to_numpy()
Y_lobes = bi.get_binary(pandas.read_csv('../../Data/CoarseGrained/Y_lobes_quan.csv'),30)
tasks_lobes = list(Y_lobes.columns)

#Define the parameters that will not vary

POP = 50
GEN = 150
FIL = 0.5
POS = 1
RAN = 1997

#Define the parameters that will vary

ALG1 = 'NSGA2'
ALG2 = 'SPEA2'
SVM1 = MonoObjectiveCV(estimator=SVC(C=2,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
SVM2 = MonoObjectiveCV(estimator=SVC(C=10,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
SVM3 = MonoObjectiveCV(estimator=SVC(C=50,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
MNB1 = MonoObjectiveCV(estimator=MultinomialNB(alpha=0.2,fit_prior=True),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
MNB2 = MonoObjectiveCV(estimator=MultinomialNB(alpha=1,fit_prior=True),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
MNB3 = MonoObjectiveCV(estimator=MultinomialNB(alpha=5,fit_prior=True),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
DTC1 = MonoObjectiveCV(estimator=DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=5,min_samples_leaf=5,class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
DTC2 = MonoObjectiveCV(estimator=DecisionTreeClassifier(criterion='gini',max_depth=15,min_samples_split=5,min_samples_leaf=5,class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
DTC3 = MonoObjectiveCV(estimator=DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=2,min_samples_leaf=2,class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
DTC4 = MonoObjectiveCV(estimator=DecisionTreeClassifier(criterion='gini',max_depth=15,min_samples_split=2,min_samples_leaf=2,class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
ES1 = MonoObjectiveCV(estimator=SVC(C=10,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),score='f1')
ES2 =  Hypervolume(estimator=SVC(C=10,kernel='rbf',cache_size=1000,gamma='scale',class_weight='balanced'),cv=StratifiedKFold(n_splits=5,shuffle=True),scores={'accuracy':1,'f1':1})
MR1 = 0.05
MR2 = 0.1
MR3 = 0.15
SE1 = TournamentSelection(k=2,replacement=False,winners=1)
SE2 = TournamentSelection(k=5,replacement=False,winners=2)
SE3 = RouletteWheel()

#Define all the possible combinations

combination_ALG = list(product(['ALG1','ALG2'],['SVM1'],['MR1'],['SE1']))
combination_SVM = list(product(['ALG1'],['SVM1','SVM2','SVM3'],['MR1'],['SE1']))
combination_MNB = list(product(['ALG1'],['MNB1','MNB2','MNB3'],['MR1'],['SE1']))
combination_DTC = list(product(['ALG1'],['DTC1','DTC2','DTC3','DTC4'],['MR1'],['SE1']))
combination_ES = list(product(['ALG1'],['ES1','ES2'],['MR1'],['SE1']))
combination_MR = list(product(['ALG1'],['SVM1'],['MR1','MR2','MR3'],['SE1']))
combination_SE = list(product(['ALG1'],['SVM1'],['MR1'],['SE1','SE2','SE3']))
sufixes = ['ALG','SVM','MNB','DTC','ES','MR','SE']

#Define and run the genetic algorithms

for sufix in sufixes:   
    combinations = globals()['combination_'+str(sufix)]   
    for task in tasks_lobes:
        try:
            y_data = Y_lobes[[task]].to_numpy().flatten()
            par = []
            for combination in combinations:
                if globals()[combination[0]]=='NSGA2':ga = NSGA2(population_size=POP,generations=GEN,fitness=globals()[combination[1]],optimize_features=True,mutation_rate=globals()[combination[2]],selection=globals()[combination[3]],positive_class=POS,random_state=RAN,id='_'.join(combination)+'_'+str(task))
                elif globals()[combination[0]]=='SPEA2': ga = SPEA2(population_size=POP,generations=GEN,fitness=globals()[combination[1]],optimize_features=True,mutation_rate=globals()[combination[2]],selection=globals()[combination[3]],positive_class=POS,random_state=RAN,id='_'.join(combination)+'_'+str(task))
                par.append(ga)
                del ga
            par_ga = Parallel(*par)
            par_ga.set_features(features)
            par_ga.fit(x_data,y_data)
            par_ga.save(dir_name='./Exploration_multi/'+str(sufix),overwrite=True)
            del par, par_ga
        except: pass
