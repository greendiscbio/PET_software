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

#Define all the possible combinations

combination_SVM = list(product(['SVM1','SVM2','SVM3'],['AN1'],['EL1'],['MR1'],['SE1']))
combination_MNB = list(product(['MNB1','MNB2','MNB3'],['AN1'],['EL1'],['MR1'],['SE1']))
combination_DTC = list(product(['DTC1','DTC2','DTC3','DTC4'],['AN1'],['EL1'],['MR1'],['SE1']))
combination_AN = list(product(['SVM1'],['AN1','AN2','AN3'],['EL1'],['MR1'],['SE1']))
combination_EL = list(product(['SVM1'],['AN1'],['EL1','EL2','EL3'],['MR1'],['SE1']))
combination_MR = list(product(['SVM1'],['AN1'],['EL1'],['MR1','MR2','MR3'],['SE1']))
combination_SE = list(product(['SVM1'],['AN1'],['EL1'],['MR1'],['SE1','SE2','SE3']))
sufixes = ['SVM','MNB','DTC','AN','EL','MR','SE']

#Define and run the genetic algorithms

for sufix in sufixes:   
    combinations = globals()['combination_'+str(sufix)]   
    for task in tasks_lobes:
        try:
            y_data = Y_lobes[[task]].to_numpy().flatten()
            par = []
            for combination in combinations:
                ga = BasicGA(population_size=POP,generations=GEN,fitness=globals()[combination[0]],annihilation=globals()[combination[1]],fill_with_elite=FIL,elitism=globals()[combination[2]],mutation_rate=globals()[combination[3]],selection=globals()[combination[4]],positive_class=POS,random_state=RAN,id='_'.join(combination)+'_'+str(task))
                par.append(ga)
                del ga
            par_ga = Parallel(*par)
            par_ga.set_features(features)
            par_ga.fit(x_data,y_data)
            par_ga.save(dir_name='./Exploration_mono/'+str(sufix),overwrite=True)
            del par, par_ga
        except: pass
