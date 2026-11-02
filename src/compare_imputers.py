"""
This program benchmarks imputation methods for perovskite solar cell data under 
empirically modeled missingness. It uses the Perovskite Database of 
Jacobsson et al. (2022), in the version preprocessed by 
Sepúlveda et al. (JCIM, 2025). Only records with a reported perovskite material
are retained, and the fully observed records are used as ground truth.

Since Little’s test rejects the MCAR assumption, the observed missingness 
is modeled with one logistic regression per frequent missingness pattern, 
using the observed variables as predictors. The resulting patterns 
(affected variables, weights and frequencies) are encoded as a multivariate 
MAR amputation mechanism in pyampute and applied repeatedly to the 
complete records to generate realistic incomplete datasets. 
Each imputation method is fitted on an amputed training set and applied, 
without refitting, to an amputed test set.

Performance is evaluated with point-wise reconstruction metrics (NRMSE, NMAE) 
and multivariate distribution-preservation metrics (energy distance, 
Wasserstein distance and the distribution preservation score) over 20 paired 
repetitions, with statistical tests comparing the methods. Finally, 
a sensitivity analysis extends the MAR mechanism to MAR+MNAR by making each 
variable’s missingness also depend on its own value, with strength controlled 
by a parameter $\gamma$ ($\gamma = 0$ corresponds to MAR).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import json
import os

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import KNNImputer

from hyperimpute.plugins.imputers import Imputers
from pyampute.exploration.mcar_statistical_tests import MCARTest

from mar_mnar import remove_outliers_iqr, preprocessing_data_prvskt
from mar_mnar import model_MAR_pattern, ampute_mar_mnar
from performance_metrics import calculate_mean_nrmse, calculate_mean_nmae, emd_wasserstein_pot
from performance_metrics import nor_energy_distance, distribution_preservation_score
from performance_metrics import organize_results

from mar_weights_table import mar_weights_to_latex


if __name__ == "__main__":
	# the variables we include in the analysis.
    var_names = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4', 'DMF_DMSO_ratio', 'first_Prvskt_annealing_temperature',
             'first_Prvskt_thermal_annealing_time', 'Perovskite_annealing_thermal_exposure', 'Perovskite_band_gap', 'Perovskite_thickness',
             'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    # outliers removal before splitting on complete and incomplete data.
    cols_to_fix = ['first_Prvskt_thermal_annealing_time', 'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    skewed_cols = ['first_Prvskt_thermal_annealing_time']    # variables with long-tails.


    # -------------------------------------hyper-parameters-------------------------------------------------------------
    threshold_pattern = 0.005                               # threshold to discard those 0.5% most rare patterns.
    R = 20                                                   # number of Monte Carlo Repetitions
    #missingness_type = 'MNAR'
    missingness_type = 'MAR'
    flag_plot = 'False'                                      # 'True': shows a plot with the missingness patterns
    show_mar_patterns  = 'False'                             # 'True': prints mar model info
    print_latex_flag = 'False'
    # ------------------------------------------------------------------------------------------------------------------


    # -------------------------------------read the data----------------------------------------------------------------
    base_dir = Path(__file__).parent.parent                                        # project location
    file_name_data = base_dir / "data" / "Descriptors_Perovskite_Database.xlsx"    # the data
    raw_data = pd.read_excel(file_name_data)
    data = raw_data[var_names].copy()
    data = data.reset_index(drop=True)                                             #- that dataset contains 41916 rows.
    # Preprocessing the data.
    data = remove_outliers_iqr(data, cols_to_fix, factor=2.5)                      # remove outliers
    for col in skewed_cols:
        data[col] = np.log1p(data[col])
    df_complete, df_incomplete = preprocessing_data_prvskt(data, var_names)  # splitting on complete and incomplete data.
    print(f"  >> Rows with complete data:   {len(df_complete)} ({len(df_complete)/(len(df_complete)+len(df_incomplete)):.1%})")
    print(f"  >> Rows with incomplete data: {len(df_incomplete)} ({len(df_incomplete)/(len(df_complete)+len(df_incomplete)):.1%})")
    #import sys
    #sys.exit("Error message")
    # ------------------------------------------------------------------------------------------------------------------


    # ----------------------------------- Little's MCAR test -----------------------------------------------------------
    mt = MCARTest(method="little")
    p_value = mt.little_mcar_test(df_incomplete)
    print(f"Little's MCAR Test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: Reject MCAR (Missingness is likely MAR or MNAR)")
    else:
        print("Conclusion: Fail to reject MCAR (Data may be missing completely at random)")
    #import sys
    #sys.exit("Error message")
    # -----------------------------------------------------------------------------------------------------------------

    
    # ------------------------------------ Estimate MAR missingness pattern --------------------------------------------
    real_prop = len(df_incomplete)/(len(df_incomplete) + len(df_complete))         # takes into accont dataset missingness prevalence
    mar_model = model_MAR_pattern(df_incomplete, threshold_pattern, real_prop, flag_plot)
    if print_latex_flag == 'True':
        latex = mar_weights_to_latex(mar_model, max_patterns=7)
        print(latex)
        mar_weights_to_latex(mar_model, weight_by_freq=True, max_patterns=20, filename="mar_weights.tex")

    if show_mar_patterns=='True':
        for i, pattern in enumerate(mar_model.patterns):
            print(f"--- pattern {i+1} ---")
            print(f"freq: {pattern['freq']}")
            print(f"mechanism: {pattern['mechanism']}")
            print(f"incomplete_vars: {pattern['incomplete_vars']}")
            print(f"weights: {pattern['weights']}\n")
    #import sys
    #ys.exit("Error message")
    #-------------------------------------------------------------------------------------------------------------------


    #------------------------------------- Compare statiscally the proportions -----------------------------------------
    X_mar = ampute_mar_mnar(df_complete, mar_model, gamma=0, seed=42)
    df_miss_raw = pd.DataFrame(X_mar, columns=df_complete.columns, index=df_complete.index)
    import sys
    sys.exit("Error message")
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------- which methods --------------------------------------------------------------
    if missingness_type == 'MAR':
        gamma_values = [0]
        the_models = [
        # ----------- The Mean Baseline
        {'estimator': DummyRegressor(strategy='mean'), 'max_iter': 1, 'name': 'Mean_Baseline'},
        {'estimator': DummyRegressor(strategy='median'), 'max_iter': 1, 'name': 'Median'},
        # -----------
        {'estimator': RandomForestRegressor(n_estimators=50, n_jobs=1, random_state=42), 'max_iter': 50,'name': 'MissForest_Classic_50'},
        {'estimator': ExtraTreesRegressor(n_estimators=50, n_jobs=1, random_state=42), 'imputation_order': 'ascending', 'max_iter': 50, 'name': 'MissForest_Extra_50'},
        {'estimator': KNeighborsRegressor(n_neighbors=5), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_KNN_5'},
        {'estimator': BayesianRidge(), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_Bayesian_Asc'},
        {'estimator': Ridge(random_state=42), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_Ridge'},
        {'estimator': Imputers().get("miwae"), 'name': 'MIWAE'},
        # ----------  Non-iterative KNN imputation ---
        {'type': 'knn', 'n_neighbors': 5,  'weights': 'uniform',  'name': 'KNN_5'}
        ]

    # models for MNAR evaluation in respect the level of missingnes 'gamma'.
    elif missingness_type == 'MNAR':
        gamma_values = [-3, -2, -1, 0, 1, 2, 3]
        the_models = [
        # --- The Mean Baseline ---
        {'estimator': DummyRegressor(strategy='mean'), 'max_iter': 1, 'name': 'Mean_Baseline'},
        {'estimator': RandomForestRegressor(n_estimators=50, n_jobs=1, random_state=42), 'max_iter': 50,'name': 'MissForest_Classic_50'},
        {'estimator': ExtraTreesRegressor(n_estimators=50, n_jobs=1, random_state=42), 'imputation_order': 'ascending', 'max_iter': 50, 'name': 'MissForest_Extra_50'},
        {'estimator': KNeighborsRegressor(n_neighbors=5), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_KNN_5'},
        {'type': 'knn', 'n_neighbors': 5,  'weights': 'uniform',  'name': 'KNN_5'}
        ]

    else:
        print('*** ERROR ON IMPUTATION TYPE *******')



    # ------------------------------------ MonteCarlo loop evaluation ---------------------------------------------------- 
    all_scores = []
    for r in range(R):                    #-- MonteCarlo: we generate enough MAR missing paterns.
        # 1. Shuffle and Split the RAW (unscaled) data
        df_complete = df_complete.sample(frac=1, random_state=r).reset_index(drop=True)
        df_train_true_raw, df_test_true_raw = train_test_split(df_complete, test_size=0.20, random_state=r)
        for gamma in gamma_values:
            # Ampute the raw data
            X_train_mar = ampute_mar_mnar(df_train_true_raw, mar_model, gamma, seed=42+r)
            df_train_miss_raw = pd.DataFrame(X_train_mar, columns=df_train_true_raw.columns, index=df_train_true_raw.index)
            X_test_mar = ampute_mar_mnar(df_test_true_raw, mar_model, gamma, seed=100+r)
            df_test_miss_raw = pd.DataFrame(X_test_mar, columns=df_test_true_raw.columns, index=df_test_true_raw.index)
            scaler = MinMaxScaler()   #scale everything for the Imputation Models
            # Fit on the raw training truth, transform the others
            df_train_true = pd.DataFrame(scaler.fit_transform(df_train_true_raw), columns=df_complete.columns)
            df_test_true  = pd.DataFrame(scaler.transform(df_test_true_raw), columns=df_complete.columns)
            # Scale the "Missing" dataframes
            df_train_miss = pd.DataFrame(scaler.transform(df_train_miss_raw), columns=df_complete.columns)
            df_test_miss  = pd.DataFrame(scaler.transform(df_test_miss_raw), columns=df_complete.columns)

            # Proceed to Imputation with df_train_miss and df_test_miss...
            for config in the_models:
                if config['name']== 'MIWAE':
                    miwae_config = {'batch_size': 64, 'n_epochs': 500, 'latent_size': 2, 'n_hidden': 128, 'K': 1}
                    miwae_model = Imputers().get("miwae", **miwae_config)
                    miwae_model.fit_transform(df_train_miss.copy())
                    df_test_imputed = miwae_model.transform(df_test_miss)
                    if hasattr(df_test_imputed, 'values'):
                        raw_values = df_test_imputed.values
                    else:
                        raw_values = df_test_imputed # It's already a numpy array
                    df_test_imputed = pd.DataFrame(raw_values, columns=df_test_true.columns, index=df_test_true.index)

                elif config['name'] == 'KNN_5':
                    imputer = KNNImputer(n_neighbors=config['n_neighbors'], weights=config['weights'])
                    imputer.fit(df_train_miss.copy())                  # donors = training rows only
                    test_imputed = imputer.transform(df_test_miss)
                    df_test_imputed = pd.DataFrame(test_imputed, columns=df_test_true.columns, index=df_test_true.index)

                else:
                    imputer = IterativeImputer(estimator=config['estimator'], max_iter=config['max_iter'], random_state=42)
                    imputer = imputer.fit(df_train_miss)
                    test_imputed  = imputer.transform(df_test_miss)
                    df_test_imputed = pd.DataFrame(test_imputed, columns=df_test_true.columns, index=df_test_true.index)
                    if config['name']== 'Mean_Baseline':
                        mean_imputed_data = df_test_imputed.copy()
                
                # calculate metrics:
                normalized_rmse = calculate_mean_nrmse(df_test_true, df_test_imputed, df_test_miss, method='std', verbose=True)
                normalized_mae  = calculate_mean_nmae(df_test_true, df_test_imputed, df_test_miss, method='std', verbose=False)
                # Force both to be numpy arrays
                array_true = np.asarray(df_test_true)
                array_imp = np.asarray(df_test_imputed)
                dist_emd_wasserstein = emd_wasserstein_pot(array_true, array_imp)    # <--- this is much faster.
                nE_dist = nor_energy_distance(df_test_true, df_test_imputed)
                rel_score = distribution_preservation_score(df_test_true, df_test_imputed, mean_imputed_data)

                # store the metrics in a dictionary
                iteration_results = {'repetition': int(r),
                        'model': config['name'],
                        'gamma': float(gamma),
                        'nrmse': float(normalized_rmse),
                        'nmae': float(normalized_mae),
                        'energy_distance': float(nE_dist),
                        'dist_preservation_score': float(rel_score),
                        'wasserstein' : float(dist_emd_wasserstein)
                        }
                all_scores.append(iteration_results)

        print(f"Repetition {r+1}/{R} complete.")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        with open(f'Scores_{missingness_type}_{R}.json', 'w') as f:
            json.dump(all_scores, f)