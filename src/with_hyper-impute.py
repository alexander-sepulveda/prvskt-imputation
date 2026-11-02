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




def tune_gain_hyperparameters(df_train_true, df_train_miss, n_trials=10):
    """
    Tunes hyperparameters for GAIN imputer using an 80/20 split.
    
    Parameters:
    df_train_true (pd.DataFrame): The complete ground truth data.
    df_train_miss (pd.DataFrame): The data with amputated/missing values.
    n_trials (int): Number of hyperparameter combinations to test.
    """

    from hyperimpute.plugins.imputers import Imputers
    import optuna
    
    # 1. Single 80/20 Split (Ensuring indices match between true and missing datasets)
    X_train_miss, X_val_miss, X_train_true, X_val_true = train_test_split(
        df_train_miss, df_train_true, test_size=0.2, random_state=42
    )
    
    # 2. Create a mask to evaluate ONLY the values that are actually missing
    # We only want to calculate the error on the cells GAIN had to guess
    val_mask = X_val_miss.isna()

    def objective(trial):
        # 3. Define the Hyperparameter Search Space
        params = {
            'n_epochs': trial.suggest_int('n_epochs', 2000, 4000, step=1000),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
        #    'alpha': trial.suggest_float('alpha', 0.5, 50.0, log=True),
            'hint_rate': trial.suggest_float('hint_rate', 0.1, 0.9)
        }
        
        # 4. Initialize the GAIN model with current trial's parameters
        imputer = Imputers().get("gain", **params)
        
        try:
            # 5. Fit on the 80% missing data, transform the 20% validation data
            imputer.fit(X_train_miss)
            X_val_imputed = imputer.transform(X_val_miss)
            
            # 6. Extract only the imputed values and their corresponding ground truth
            true_vals = X_val_true[val_mask].values
            pred_vals = X_val_imputed[val_mask].values
            
            # Calculate Root Mean Squared Error (RMSE)
            rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
            # If the math resulted in NaN, fail gracefully
            if np.isnan(rmse):
                return float('inf')
            
            return rmse
            
        except Exception as e:
            # GANs can occasionally crash or diverge with bad hyperparameters.
            # If this trial fails, return infinity so Optuna knows it's a bad combination.
            return float('inf')

    # 7. Run the Optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Keeps console output clean
    print(f"Starting hyperparameter search with {n_trials} trials...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("\n--- Tuning Complete ---")
    print(f"Best Validation RMSE: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    return study.best_params





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
    # ------------------------------------------------------------------------------------------------------------------


    # ----------------------------------- Little's MCAR test -----------------------------------------------------------
    mt = MCARTest(method="little")
    p_value = mt.little_mcar_test(df_incomplete)
    print(f"Little's MCAR Test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: Reject MCAR (Missingness is likely MAR or MNAR)")
    else:
        print("Conclusion: Fail to reject MCAR (Data may be missing completely at random)")
    # -----------------------------------------------------------------------------------------------------------------

    
    # ------------------------------------ Estimate MAR missingness pattern --------------------------------------------
    real_prop = len(df_incomplete)/(len(df_incomplete) + len(df_complete))         # takes into accont dataset missingness prevalence
    mar_model = model_MAR_pattern(df_incomplete, threshold_pattern, real_prop, flag_plot)
    if show_mar_patterns=='True':
        for i, pattern in enumerate(mar_model.patterns):
            print(f"--- pattern {i+1} ---")
            print(f"freq: {pattern['freq']}")
            print(f"mechanism: {pattern['mechanism']}")
            print(f"incomplete_vars: {pattern['incomplete_vars']}")
            print(f"weights: {pattern['weights']}\n")
    #-------------------------------------------------------------------------------------------------------------------


    #------------------------------------- Compare statiscally the proportions -----------------------------------------
    X_mar = ampute_mar_mnar(df_complete, mar_model, gamma=0, seed=42)
    df_miss_raw = pd.DataFrame(X_mar, columns=df_complete.columns, index=df_complete.index)
    #import sys
    #sys.exit("Error message")
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------- which methods --------------------------------------------------------------
    if missingness_type == 'MAR':
        gamma_values = [0]
        the_models = [
        # ----------- The Mean Baseline
        {'estimator': DummyRegressor(strategy='mean'), 'max_iter': 1, 'name': 'Mean_Baseline'},
        #{'estimator': Imputers().get("gain"), 'name': 'HyperImpute_GAIN'}
        {'estimator': Imputers().get("hyperimpute"), 'name': 'hyper-impute'}
        # -----------

        ]

    # models for MNAR evaluation in respect the level of missingnes 'gamma'.
    elif missingness_type == 'MNAR':
        gamma_values = [-3, -2, -1, 0, 1, 2, 3]
        the_models = [
        # --- The Mean Baseline ---
        {'estimator': DummyRegressor(strategy='mean'), 'max_iter': 1, 'name': 'Mean_Baseline'},
        {'estimator': RandomForestRegressor(n_estimators=50, n_jobs=1, random_state=42), 'max_iter': 50,'name': 'MissForest_Classic_50'},
        {'estimator': ExtraTreesRegressor(n_estimators=50, n_jobs=1, random_state=42), 'imputation_order': 'ascending', 'max_iter': 50, 'name': 'MissForest_Extra_50'},
        {'estimator': KNeighborsRegressor(n_neighbors=5), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_KNN_5'}        
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
            # 2. Ampute the RAW data (This is where the MAR logic works best!)
            X_train_mar = ampute_mar_mnar(df_train_true_raw, mar_model, gamma, seed=42+r)
            #X_train_mar = mar_model.transform(df_train_true_raw)
            df_train_miss_raw = pd.DataFrame(X_train_mar, columns=df_train_true_raw.columns, index=df_train_true_raw.index)
            X_test_mar = ampute_mar_mnar(df_test_true_raw, mar_model, gamma, seed=100+r)
            #X_test_mar = mar_model.transform(df_test_true_raw)
            df_test_miss_raw = pd.DataFrame(X_test_mar, columns=df_test_true_raw.columns, index=df_test_true_raw.index)
            # 3. NOW Scale everything for the Imputation Models
            scaler = MinMaxScaler()
            # Fit on the RAW training truth, transform the others
            df_train_true = pd.DataFrame(scaler.fit_transform(df_train_true_raw), columns=df_complete.columns)
            df_test_true  = pd.DataFrame(scaler.transform(df_test_true_raw), columns=df_complete.columns)
            # Scale the "Missing" dataframes (MinMaxScaler ignores NaNs automatically)
            df_train_miss = pd.DataFrame(scaler.transform(df_train_miss_raw), columns=df_complete.columns)
            df_test_miss  = pd.DataFrame(scaler.transform(df_test_miss_raw), columns=df_complete.columns)
            # 4. Proceed to Imputation with df_train_miss and df_test_miss...

            for config in the_models:
                if config['name'] == 'hyper-impute':
                    plugin = config['estimator']
                    plugin.fit_transform(df_train_miss.copy())
                    df_test_imputed = plugin.transform(df_test_miss)
                    if hasattr(df_test_imputed, 'values'):
                        raw_values = df_test_imputed.values
                    else:
                        raw_values = df_test_imputed # It's already a numpy array
                    df_test_imputed = pd.DataFrame(raw_values, columns=df_test_true.columns, index=df_test_true.index)

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

                # STORE ALL METRICS in a dictionary for this repetition
                iteration_results = {'repetition': int(r),
                        'model': config['name'],
                        'gamma': float(gamma),
                        'nrmse': float(normalized_rmse),
                        'nmae': float(normalized_mae),
                        'energy_distance': float(nE_dist),
                        'dist_preservation_score': float(rel_score),
                        'wasserstein' : float(dist_emd_wasserstein)
                        }
                print(iteration_results)
                #all_scores[config['name']].append(iteration_results)
                all_scores.append(iteration_results)

        print(f"Repetition {r+1}/{R} complete.")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        with open('scores_MAR_hyper-impute.json', 'w') as f:
            json.dump(all_scores, f)