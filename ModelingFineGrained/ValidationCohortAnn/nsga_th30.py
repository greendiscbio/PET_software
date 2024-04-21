import sys
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from datetime import datetime

# https://github.com/FernandoGaGu/GoJo-ML
sys.path.append('/home/fgarcia/github/GoJo-ML')

# https://github.com/FernandoGaGu/pywinEA2
sys.path.append('/home/fgarcia/github/pywinEA2')

import gojo
import pywinEA2



# processed data from ../missing_values/Analysis and processing - vX.ipynb
DATA_FILE = os.path.join('..', 'data', 'final_data_MICE.parquet')

# processed data from ../data_processing (validation cohort)
VAL_DATA_FILE = os.path.join('..', 'data', 'final_data_MICE_val_cohort_v1.parquet')

# input variables
INPUT_VARIABLES = [
    'diagnosis_AD',
    'diagnosis_HC',
    'diagnosis_bvFTD',
    'sex',
    'age',
    'years_of_formal_education',

    'Digit span forward',
    'Digit span backward',
    'Corsi’s test forward',
    'Corsi’s test backward',
    'TMT A',
    'TMT B',
    'SDMT',
    'SCWIT word reading',
    'SCWIT color naming',
    'SCWIT interference',
    'ToL correct',
    'ToL movements',
    'ToL start',
    'ToL execution',
    'ToL resolution',
    'BNT',
    'Semantic fluency',
    'Letter fluency',
    'FCSRT free recall 1',
    'FCSRT total free recall',
    'FCSRT total recall',
    'FCSRT delayed free recall',
    'FCSRT delayed total recall',
    'ROCF copy accuracy',
    'ROCF memory 3 min',
    'ROCF memory 30 min',
    'ROCF time',
    'ROCF recognition',
    'JLO',
    'VOSP object decision',
    'VOSP progressive silhouettes',
    'VOSP discrimination of position',
    'VOSP number location',
]

# variables to standarize (everything except sex and diagnosis)
VAR_TO_STANDARIZE = [
    'age',
    'years_of_formal_education',
    'Digit span forward',
    'Digit span backward',
    'Corsi’s test forward',
    'Corsi’s test backward',
    'TMT A',
    'TMT B',
    'SDMT',
    'SCWIT word reading',
    'SCWIT color naming',
    'SCWIT interference',
    'ToL correct',
    'ToL movements',
    'ToL start',
    'ToL execution',
    'ToL resolution',
    'BNT',
    'Semantic fluency',
    'Letter fluency',
    'FCSRT free recall 1',
    'FCSRT total free recall',
    'FCSRT total recall',
    'FCSRT delayed free recall',
    'FCSRT delayed total recall',
    'ROCF copy accuracy',
    'ROCF memory 3 min',
    'ROCF memory 30 min',
    'ROCF time',
    'ROCF recognition',
    'JLO',
    'VOSP object decision',
    'VOSP progressive silhouettes',
    'VOSP discrimination of position',
    'VOSP number location',
]

# target variables
TARGET_VARIABLES = [
    'precentral_l',
    'precentral_r',
    'postcentral_l',
    'postcentral_r',
    'rolandic_oper_l',
    'rolandic_oper_r',
    'frontal_sup_l',
    'frontal_sup_r',
    'frontal_mid_l',
    'frontal_mid_r',
    'frontal_inf_oper_l',
    'frontal_inf_oper_r',
    'frontal_inf_tri_l',
    'frontal_inf_tri_r',
    'frontal_sup_medial_l',
    'frontal_sup_medial_r',
    'supp_motor_area_l',
    'supp_motor_area_r',
    'paracentral_lobule_l',
    'paracentral_lobule_r',
    'frontal_sup_orb_l',
    'frontal_sup_orb_r',
    'frontal_med_orb_l',
    'frontal_med_orb_r',
    'frontal_mid_orb_l',
    'frontal_mid_orb_r',
    'frontal_inf_orb_l',
    'frontal_inf_orb_r',
    'rectus_l',
    'rectus_r',
    'olfactory_l',
    'olfactory_r',
    'temporal_sup_l',
    'temporal_sup_r',
    'heschl_l',
    'heschl_r',
    'temporal_mid_l',
    'temporal_mid_r',
    'temporal_inf_l',
    'temporal_inf_r',
    'parietal_sup_l',
    'parietal_sup_r',
    'parietal_inf_l',
    'parietal_inf_r',
    'angular_l',
    'angular_r',
    'supramarginal_l',
    'supramarginal_r',
    'precuneus_l',
    'precuneus_r',
    'occipital_sup_l',
    'occipital_sup_r',
    'occipital_mid_l',
    'occipital_mid_r',
    'occipital_inf_l',
    'occipital_inf_r',
    'cuneus_l',
    'cuneus_r',
    'calcarine_l',
    'calcarine_r',
    'lingual_l',
    'lingual_r',
    'fusiform_l',
    'fusiform_r',
    'temporal_pole_sup_l',
    'temporal_pole_sup_r',
    'temporal_pole_mid_l',
    'temporal_pole_mid_r',
    'cingulum_ant_l',
    'cingulum_ant_r',
    'cingulum_mid_l',
    'cingulum_mid_r',
    'cingulum_post_l',
    'cingulum_post_r',
    'hippocampus_l',
    'hippocampus_r',
    'parahippocampal_l',
    'parahippocampal_r',
    'insula_l',
    'insula_r',
    'amygdala_l',
    'amygdala_r',
    'caudate_l',
    'caudate_r',
    'putamen_l',
    'putamen_r',
    'pallidum_l',
    'pallidum_r',
    'thalamus_l',
    'thalamus_r',
]

# variables used to train a baseline model
BASELINE_MODEL_VARIABLES = [
    'diagnosis_AD',
    'diagnosis_HC',
    'diagnosis_bvFTD',
    'sex',
    'age',
    'years_of_formal_education',
]


# hypometabolism threshold
THRESHOLD = 30

# minimum proportion of hypometabolic voxels
MINIMUM_PROP = 0.05

# best-model definition
MODEL = SVC
MODEL_PARAMS = dict(
    kernel='rbf',
    C=2,
    gamma='scale',
    class_weight={0: 1, 1: 40},
    random_state=1997,
    cache_size=2000
)

# genetic algorithm hyperparameters
GA_PARAMS = dict(
    population_size=100,
    max_generations=300,
    p_crossover=0.75,
    p_mutation=0.25,
    selection_op='tournament',
)

# Number of times to simulate the introduction of random errors diagnostic errors into the model
N_ERROR_SIMULATIONS = 100

# version
VERSION = 'v1'


def simulate_diagnostic_errors(X: pd.DataFrame, n_iter: int):
    """ Subroutine used to introduce diagnostic errors as described in the article:  
    https://doi.org/10.1002/gps.5667
    """
    X = X.copy()  # avoid inplace modifications
    
    # transition probabilities (based on Figure 5)
    p_ad_to_hc  = 12 / 170
    p_ad_to_ftd = 17 / 170
    p_hc_to_ad  = 8 / 87
    p_hc_to_ftd = 4 / 87
    p_ftd_to_ad = 10 / 72
    p_ftd_to_hc = 3 / 72
    
    # create repetitions of the same dataframe
    X = pd.concat([X.copy() for _ in range(n_iter)], ignore_index=True)
    
    # save rows that have been modified
    mod_mask = np.zeros(shape=(X.shape[0]), dtype=bool)
    
    # apply HC error introduction
    if 'diagnosis_HC' in X.columns:
        # use a uniform distribution to modify the values according to the probabilities of diagnostic error
        sample_probs = np.random.uniform(size=X.shape[0])
        
        # mask values that have already been changed
        sample_probs[mod_mask] = np.inf
        
        # apply HC to AD diagnosis change
        hc_to_ad_mask = (sample_probs <= p_hc_to_ad) & (X['diagnosis_HC'].values == 1)
        X.loc[hc_to_ad_mask, 'diagnosis_HC'] = 0
        
        if 'diagnosis_AD' in X.columns:
            X.loc[hc_to_ad_mask, 'diagnosis_AD'] = 1
        
        # save changes in the modification mask
        mod_mask[hc_to_ad_mask] = True
        
        # (2) use a uniform distribution to modify the values according to the probabilities of diagnostic error
        sample_probs = np.random.uniform(size=X.shape[0])
        
        # mask values that have already been changed
        sample_probs[mod_mask] = np.inf
        
        # apply HC to FTD diagnosis change
        hc_to_ftd_mask = (sample_probs <= p_hc_to_ftd) & (X['diagnosis_HC'].values == 1)
        X.loc[hc_to_ftd_mask, 'diagnosis_HC'] = 0
        
        if 'diagnosis_bvFTD' in X.columns:
            X.loc[hc_to_ftd_mask, 'diagnosis_bvFTD'] = 1
        
        # save changes in the modification mask
        mod_mask[hc_to_ftd_mask] = True
        

    # apply  AD error introduction
    if 'diagnosis_AD' in X.columns:
        # use a uniform distribution to modify the values according to the probabilities of diagnostic error
        sample_probs = np.random.uniform(size=X.shape[0])
        
        # mask values that have already been changed
        sample_probs[mod_mask] = np.inf
        
        # apply AD to HC diagnosis change
        ad_to_hc_mask = (sample_probs <= p_ad_to_hc) & (X['diagnosis_AD'].values == 1)
        X.loc[ad_to_hc_mask, 'diagnosis_AD'] = 0
        
        if 'diagnosis_HC' in X.columns:
            X.loc[ad_to_hc_mask, 'diagnosis_HC'] = 1
        
        # save changes in the modification mask
        mod_mask[ad_to_hc_mask] = True
        
        # (2) use a uniform distribution to modify the values according to the probabilities of diagnostic error
        sample_probs = np.random.uniform(size=X.shape[0])
        
        # mask values that have already been changed
        sample_probs[mod_mask] = np.inf
        
        # apply AD to FTD diagnosis change
        ad_to_ftd_mask = (sample_probs <= p_ad_to_ftd) & (X['diagnosis_AD'].values == 1)
        X.loc[ad_to_ftd_mask, 'diagnosis_AD'] = 0
        
        if 'diagnosis_bvFTD' in X.columns:
            X.loc[ad_to_ftd_mask, 'diagnosis_bvFTD'] = 1
        
        # save changes in the modification mask
        mod_mask[ad_to_ftd_mask] = True
        
        
    # apply FTD error introduction
    if 'diagnosis_bvFTD' in X.columns:
        # use a uniform distribution to modify the values according to the probabilities of diagnostic error
        sample_probs = np.random.uniform(size=X.shape[0])
        
        # mask values that have already been changed
        sample_probs[mod_mask] = np.inf
        
        # apply FTD to AD diagnosis change
        ftd_to_ad_mask = (sample_probs <= p_ftd_to_ad) & (X['diagnosis_bvFTD'].values == 1)
        X.loc[ftd_to_ad_mask, 'diagnosis_bvFTD'] = 0
        
        if 'diagnosis_AD' in X.columns:
            X.loc[ftd_to_ad_mask, 'diagnosis_AD'] = 1
        
        # save changes in the modification mask
        mod_mask[ftd_to_ad_mask] = True
        
        # (2) use a uniform distribution to modify the values according to the probabilities of diagnostic error
        sample_probs = np.random.uniform(size=X.shape[0])
        
        # mask values that have already been changed
        sample_probs[mod_mask] = np.inf
        
        # apply FTD to HC diagnosis change
        ftd_to_hc_mask = (sample_probs <= p_ftd_to_hc) & (X['diagnosis_bvFTD'].values == 1)
        X.loc[ftd_to_hc_mask, 'diagnosis_bvFTD'] = 0
        
        if 'diagnosis_HC' in X.columns:
            X.loc[ftd_to_hc_mask, 'diagnosis_HC'] = 1
        
        # save changes in the modification mask
        mod_mask[ftd_to_hc_mask] = True
        
    print('Number of values changed: {} / {} ({:.2f}%)'.format(
        mod_mask.sum(), 
        mod_mask.shape[0], 
        mod_mask.sum() / mod_mask.shape[0] * 100))
    
    return X



# load the data
data = pd.read_parquet(DATA_FILE)

# binarize metabolism data
data[TARGET_VARIABLES] = (data[TARGET_VARIABLES] > THRESHOLD).astype(int)

# remove regions that doesn't reach the established hypometabolism threshold
hypometabolism_props = (data[TARGET_VARIABLES].sum() / data.shape[0])
rois_to_remove = list(hypometabolism_props.loc[hypometabolism_props < MINIMUM_PROP].index)

print('%d ROIs will be removed: %r' % (len(rois_to_remove), rois_to_remove))

TARGET_VARIABLES = list(filter(lambda x: x not in rois_to_remove, TARGET_VARIABLES))

# separate input and target data
X, y = data[INPUT_VARIABLES], data[TARGET_VARIABLES]

# convert X data to z-scores and save statistics (for some variables)
scaler = StandardScaler().fit(X[VAR_TO_STANDARIZE])
X.loc[:, VAR_TO_STANDARIZE] = scaler.transform(X[VAR_TO_STANDARIZE])


validation_data = pd.read_parquet(VAL_DATA_FILE)

# fix sex variable
validation_data['sex'] = validation_data['sex'] - 1

# binarize metabolism data
validation_data[TARGET_VARIABLES] = (validation_data[TARGET_VARIABLES] > THRESHOLD).astype(int)

# separate input and target data
validation_X, validation_y = validation_data[INPUT_VARIABLES], validation_data[TARGET_VARIABLES]

validation_X.loc[:, VAR_TO_STANDARIZE] = scaler.transform(validation_X[VAR_TO_STANDARIZE])

validation_X.shape, validation_y.shape



# test the model with cross-validation
cross_validation_results = []
selected_features_per_roi = {}

for i, var in enumerate(TARGET_VARIABLES):
    try:
        print('ROI "%s" (%d / %d)\n' % (var, i+1, len(TARGET_VARIABLES)))

        # model initialization
        model = MODEL(**MODEL_PARAMS)

        # GA initialization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ga = pywinEA2.MultiObjFeatureSelectionNSGA2(
                model=model,
                data=pd.concat([X, y[[var]]], axis=1),
                score='f1',
                optim='max',
                cv=5,
                cv_reps=1,
                stratified=True,
                target_feats=X.columns.tolist(),
                y=[var],
                n_jobs=1,
                **GA_PARAMS
            )

            # run the GA
            print('Performing feature selection...')

            ga_report = pywinEA2.run(ga, type='nsga2', verbose=False)

        # get the GA fitness history
        ga_fitness_history = np.array(ga_report.get('multiobj_fitness_values_max'))[:, 0]

        print('F1 score optimized from %.3f to %.3f' % (ga_fitness_history[0], ga_fitness_history[-1]))

        # get the best combination of featuresd
        pareto_front = ga_report.pareto_front
        fitness = np.array([ind.fitness.values for ind in pareto_front])
        selected_features = X.columns[np.array(pareto_front[np.argmax(fitness[:, 0])], dtype=bool)].tolist()

        # save selected features
        selected_features_per_roi[var] = selected_features

        print('\nSelected features:\n%s' % ' - '.join(list(map(lambda v: '"%s"' % v, selected_features))))

        # display GA convergence
        #ga_report.displayMultiObjectiveConvergence(title='Convergence', objective_names=['f1', 'Features'], figsize=(5, 3))

        # ============= re-evaluate the model by using a cross-validation
        cv_report = gojo.core.evalCrossVal(
            X=X[selected_features].values,
            y=y[var].values,
            model=gojo.core.SklearnModelWrapper(
                model_class=MODEL,
                **MODEL_PARAMS
            ),
            cv=gojo.util.splitter.getCrossValObj(
                cv=10, repeats=5, stratified=True, random_state=1997),
            save_train_preds=True,
            save_models=False,
            n_jobs=10
        )

        # calculate cross-validation metrics
        cv_scores = cv_report.getScores(
            gojo.core.getDefaultMetrics(
                'binary_classification', bin_threshold=0.5,
                select=['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score']
            ))
        cv_results = pd.concat([
            pd.DataFrame(cv_scores['train'].mean(axis=0)).round(decimals=3),
            pd.DataFrame(cv_scores['test'].mean(axis=0)).round(decimals=3)],
            axis=1).drop(index=['n_fold'])
        cv_results.columns = ['Train', 'Test']

        # format results
        cv_results = cv_results.T
        cv_results.index.names = ['set']
        cv_results['ROI'] = var
        cv_results['key'] = 'SVM-retrained-CV'
        cv_results = cv_results.reset_index()

        cross_validation_results.append(cv_results)


        # ============= evaluate a base model using only demographic and diagnosis
        base_report = gojo.core.evalCrossVal(
            X=X[BASELINE_MODEL_VARIABLES].values,
            y=y[var].values,
            model=gojo.core.SklearnModelWrapper(
                model_class=MODEL,
                **MODEL_PARAMS
            ),
            cv=gojo.util.splitter.getCrossValObj(
                cv=10, repeats=5, stratified=True, random_state=1997),
            save_train_preds=True,
            save_models=False,
            n_jobs=10
        )

        # calculate cross-validation metrics
        base_scores = base_report.getScores(
            gojo.core.getDefaultMetrics(
                'binary_classification', bin_threshold=0.5,
                select=['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score']
            ))
        base_results = pd.concat([
            pd.DataFrame(base_scores['train'].mean(axis=0)).round(decimals=3),
            pd.DataFrame(base_scores['test'].mean(axis=0)).round(decimals=3)],
            axis=1).drop(index=['n_fold'])
        base_results.columns = ['Train', 'Test']

        # format results
        base_results = base_results.T
        base_results.index.names = ['set']
        base_results['ROI'] = var
        base_results['key'] = 'SVM-base-CV'
        base_results = base_results.reset_index()

        cross_validation_results.append(base_results)

        # ============= evaluate the final model over the validation cohort
        # make predictions simulating diagnostic errors
        fitted_model = model.fit(X[selected_features], y[var])
        y_pred_validation = fitted_model.predict(
            simulate_diagnostic_errors(validation_X[selected_features], N_ERROR_SIMULATIONS))
        y_true_validation = np.tile(validation_y[var].values, N_ERROR_SIMULATIONS)
        assert y_pred_validation.shape[0] == y_true_validation.shape[0]

        validation_scores = pd.DataFrame([gojo.core.getScores(
            y_true=y_true_validation,
            y_pred=y_pred_validation,
            metrics=gojo.core.getDefaultMetrics(
                'binary_classification', bin_threshold=0.5,
                select=['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score']
            )
        )])
        validation_scores['set'] = 'Validation'
        validation_scores['ROI'] = var
        validation_scores['key'] = 'SVM-retrained-CV'

        validation_scores = validation_scores[['set',
         'accuracy',
         'precision',
         'recall',
         'sensitivity',
         'specificity',
         'f1_score',
         'ROI',
         'key']]

        cross_validation_results.append(validation_scores)


        # ============= evaluate the final model over the validation cohort using only diagnostic and demographics
        # make predictions simulating diagnostic errors
        fitted_model_base = model.fit(X[BASELINE_MODEL_VARIABLES], y[var])
        y_pred_validation_base = fitted_model_base.predict(
            simulate_diagnostic_errors(validation_X[BASELINE_MODEL_VARIABLES], N_ERROR_SIMULATIONS))
        y_true_validation_base = np.tile(validation_y[var].values, N_ERROR_SIMULATIONS)
        assert y_pred_validation_base.shape[0] == y_true_validation_base.shape[0]

        validation_scores_base = pd.DataFrame([gojo.core.getScores(
            y_true=y_true_validation_base,
            y_pred=y_pred_validation_base,
            metrics=gojo.core.getDefaultMetrics(
                'binary_classification', bin_threshold=0.5,
                select=['accuracy', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score']
            )
        )])
        validation_scores_base['set'] = 'Validation'
        validation_scores_base['ROI'] = var
        validation_scores_base['key'] = 'SVM-base-CV'

        validation_scores_base = validation_scores_base[['set',
         'accuracy',
         'precision',
         'recall',
         'sensitivity',
         'specificity',
         'f1_score',
         'ROI',
         'key']]

        cross_validation_results.append(validation_scores_base)

        print('\n')

    except Exception as ex:
        print('Exception in ROI "{}". Exception: {}'.format(var, ex))
        continue

cross_validation_results_df = pd.concat(cross_validation_results, axis=0)

# export results
cross_validation_results_df.to_parquet(
    os.path.join(
        '..', 'results',
        '%s_results_cv_GA_th%d_%s.parquet' % (datetime.now().strftime('%Y%m%d'), THRESHOLD, VERSION)
    ))

# save the selected features
selected_features_df = pd.DataFrame(
    index=TARGET_VARIABLES,
    columns=X.columns)

for roi, features in selected_features_per_roi.items():
    selected_features_df.loc[roi, :] = 0
    selected_features_df.loc[roi, features] = 1

selected_features_df.to_parquet(
    os.path.join(
        '..', 'results',
        '%s_selected_features_GA_th%d_%s.parquet' % (datetime.now().strftime('%Y%m%d'), THRESHOLD, VERSION)
    ))
