# In this program we compare several strategies for data imputation in a MAR scenario.
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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

import ot
import dcor

from MAR import preprocessing_data_prvskt
from MAR import model_MAR_pattern


def remove_outliers_iqr(df, columns, factor=2.5):
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        stat_lower = Q1 - (factor * IQR)
        # If the variable is a 'time', we usually want 0 to be the absolute floor
        lower_bound = max(0, stat_lower) 
        upper_bound = Q3 + (factor * IQR)
        
        # Now 0 values will be kept because 0 >= 0 is True
        mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        df_clean = df_clean[mask]

    return df_clean


def calculate_mean_nrmse(df_true, df_imputed, df_miss, method='std', verbose=False):
    """
    Calculates the average Normalized Root Mean Squared Error across all imputed columns.
    Only compares values at the locations where data was originally missing.
    
    Parameters:
    -----------
    df_true : pd.DataFrame
        The ground truth dataset (complete, no NaNs).
    df_imputed : pd.DataFrame
        The dataset after applying your imputation algorithm.
    df_miss : pd.DataFrame
        The incomplete dataset containing NaNs (output of the MAR simulation).
    method : str, 'std' or 'min_max'
        The scaling factor for normalization. 
        Note: If data is already 0-1 scaled, 'min_max' will effectively return raw RMSE.
    verbose : bool
        If True, prints the NRMSE for each individual column.
    """
    
    # Ensure df_miss is a DataFrame for consistent iteration
    if isinstance(df_miss, pd.Series):
        df_miss = df_miss.to_frame()

    column_nrmses = {}
    
    for col in df_miss.columns:
        # Safety check: ensures the column exists in all three dataframes
        if col not in df_true.columns or col not in df_imputed.columns:
            continue
            
        # 1. Create mask based on the NaN positions in the 'broken' dataframe
        mask = df_miss[col].isnull().to_numpy()
        
        if mask.any():
            # 2. Extract the true values and the imputed values only for the "holes"
            #y_true = df_true.loc[mask, col]
            y_true = df_true[col].to_numpy()[mask]
            y_pred = df_imputed[col].to_numpy()[mask]
            #y_pred = df_imputed.loc[mask, col]
            
            # 3. Calculate raw RMSE for this column
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # 4. Calculate the scaling factor (the denominator)
            # We use y_true (the subset) to normalize relative to the specific missing data range
            if method == 'std':
                scale = np.std(y_true)
                #scale = y_true.std()
            elif method == 'min_max':
                scale = np.ptp(y_true)
                #scale = y_true.max() - y_true.min()
            else:
                scale = 1.0  # Returns raw RMSE
                
            # 5. Handle potential zero-division (e.g., if y_true is constant)
            if scale > 1e-9:
                column_nrmses[col] = rmse / scale
            else:
                # If the true values being compared are constant, division is impossible.
                # We return the RMSE as is, which should be 0 if the model guessed correctly.
                column_nrmses[col] = rmse 
                
    # Verbose output for debugging
    if verbose and column_nrmses:
        print("\n--- Individual Column NRMSE ---")
        for c, score in column_nrmses.items():
            print(f"{c:20}: {score:.6f}")
            
    # Calculate the average across all columns that had missing data
    scores_list = list(column_nrmses.values())
    
    if not scores_list:
        if verbose: print("No missing values found to evaluate.")
        return 0.0
        
    return np.mean(scores_list)



def calculate_mean_nmae(df_true, df_imputed, df_miss, method='std', verbose=False):
    """
    Calculates the average Normalized Mean Absolute Error across all imputed columns.
    Only compares values at the locations where data was originally missing.
    
    Parameters:
    -----------
    df_true : pd.DataFrame
        The ground truth dataset (complete, no NaNs).
    df_imputed : pd.DataFrame
        The dataset after applying your imputation algorithm.
    df_miss : pd.DataFrame
        The incomplete dataset containing NaNs (output of the MAR simulation).
    method : str, 'std' or 'min_max'
        The scaling factor for normalization.
    verbose : bool
        If True, prints the NMAE for each individual column.
    """
    
    # Ensure df_miss is a DataFrame
    if isinstance(df_miss, pd.Series):
        df_miss = df_miss.to_frame()

    column_nmaes = {}
    
    for col in df_miss.columns:
        # Check column existence
        if col not in df_true.columns or col not in df_imputed.columns:
            continue
            
        # 1. Identify where the MAR model removed data
        mask = df_miss[col].isnull().to_numpy()
        
        if mask.any():
            # Extract subset for the "holes"
            y_true = df_true[col].to_numpy()[mask]
            y_pred = df_imputed[col].to_numpy()[mask]

            mae = mean_absolute_error(y_true, y_pred)
            
            # Using the subset y_true for normalization
            if method == 'std':
                #scale = y_true.std()
                scale = np.std(y_true)
            elif method == 'min_max':
                scale = np.ptp(y_true)
                #scale = y_true.max() - y_true.min()
            else:
                scale = 1.0  # Returns raw MAE
                
            # 5. Handle potential zero-division
            if scale > 1e-9:
                column_nmaes[col] = mae / scale
            else:
                # If scale is 0, return the raw MAE
                column_nmaes[col] = mae 
                
    if verbose and column_nmaes:
        print("\n--- Individual Column NMAE ---")
        for c, score in column_nmaes.items():
            print(f"{c:20}: {score:.6f}")
            
    scores_list = list(column_nmaes.values())
    
    if not scores_list:
        if verbose: print("No missing values found to evaluate.")
        return 0.0
        
    return np.mean(scores_list)


def organize_results(all_scores, file_name):
    # Dynamically find the number of repetitions (R) from the first list
    R = len(list(all_scores.values())[0])
    # We will store the summary data here to make a nice DataFrame at the end
    summary_data = []
    for model_name, scores in all_scores.items():
        # 1. Calculate Mean and Sample Standard Deviation
        mean_nrmse = np.mean(scores)
        std_nrmse = np.std(scores, ddof=1) # ddof=1 for sample standard deviation
        # 2. Calculate 95% Confidence Interval
        # Formula: Mean ± Z * (StdDev / sqrt(N)) ... where Z = 1.96 for 95% confidence
        margin_of_error = 1.96 * (std_nrmse / np.sqrt(R))
        ci_lower = mean_nrmse - margin_of_error
        ci_upper = mean_nrmse + margin_of_error

        # 3. Print the results to the console
        print(f"🔹 Configuration: {model_name}")
        print(f"   Mean NRMSE:  {mean_nrmse:.4f}")
        print(f"   Std Dev:     {std_nrmse:.4f}")
        print(f"   95% CI:      [{ci_lower:.4f}, {ci_upper:.4f}]")
        print("-" * 50)

        # 4. Save to our summary list
        summary_data.append({
            'Model': model_name,
            'Mean_NRMSE': round(mean_nrmse, 4),
            'Std_Dev': round(std_nrmse, 4),
            'CI_Lower': round(ci_lower, 4),
            'CI_Upper': round(ci_upper, 4)
            })

    # Create a clean tabular DataFrame
    df_results = pd.DataFrame(summary_data)
    # Sort by Mean NRMSE so the "winner" is at the top!
    df_results = df_results.sort_values(by='Mean_NRMSE', ascending=True).reset_index(drop=True)
    print("\n🏆 LEADERBOARD (Lowest NRMSE is best):")
    print(df_results.to_string(index=False))

    # Optional: Save to CSV for your paper
    df_results.to_csv(file_name, index=False)

    return 0


def emd_wasserstein_pot(true_scaled, imp_scaled):
    # Create uniform weights for each data point (all points matter equally)
    n = len(true_scaled)
    a, b = np.ones((n,)) / n, np.ones((n,)) / n  
    
    # Calculate the cost matrix (Euclidean distances between all pairs of points)
    # ot.dist computes the squared Euclidean distance by default
    M = ot.dist(true_scaled, imp_scaled, metric='euclidean')
    
    # Solve the exact Earth Mover's Distance (Wasserstein)
    w_distance = ot.emd2(a, b, M)
    
    return w_distance


def normalized_energy_distance(true_scaled, imp_scaled):
    # Calculate raw energy distance. It assumes min_max scaling has been applied.
    raw_ed = dcor.energy_distance(true_scaled, imp_scaled)
    N = true_scaled.shape[1]
    max_possible_ed = 2 * np.sqrt(N)  # Calculate the theoretical maximum
    E_dist = raw_ed / max_possible_ed  # Standardize to [0, 1]

    return E_dist

def distribution_preservation_score(true_scaled, imp_model_scaled, imp_mean_scaled):
    baseline_dist = dcor.energy_distance(true_scaled, imp_mean_scaled)    # "dumb" mean baseline distance
    print('>>>> baseline distance')
    print(baseline_dist)
    model_dist = dcor.energy_distance(true_scaled, imp_model_scaled)
    print('>>>> model distance')
    print(model_dist)
    score = 1 - (model_dist / baseline_dist)

    return score



if __name__ == "__main__":
	# ----- hyper-parameters ---
	# the variables we include in the analysis.
    var_names = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4', 'DMF_DMSO_ratio', 'first_Prvskt_annealing_temperature',
             'first_Prvskt_thermal_annealing_time', 'Perovskite_annealing_thermal_exposure', 'Perovskite_band_gap', 'Perovskite_thickness',
             'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    # outliers removal before splitting on complete and incomplete data.
    cols_to_fix = ['first_Prvskt_thermal_annealing_time', 'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    skewed_cols = ['first_Prvskt_thermal_annealing_time']    # variables with long-tails.
    threshold_pattern = 0.01                                 # threshold to discard those 1% rare patterns.
    R = 30                                                   # Monte Carlo Repetitions
    # -------------------------

    # --- read the data ------
    base_dir = Path(__file__).parent.parent                                        # project location
    file_name_data = base_dir / "data" / "Descriptors_Perovskite_Database.xlsx"    # the data
    raw_data = pd.read_excel(file_name_data)
    data = raw_data[var_names].copy()
    data = data.reset_index(drop=True)                                       # that dataset contains 41197 rows.
    # Preprocessing the data.
    data = remove_outliers_iqr(data, cols_to_fix, factor=2.5)     
    
    for col in skewed_cols:
        data[col] = np.log1p(data[col])

    df_complete, df_incomplete = preprocessing_data_prvskt(data, var_names)  # splitting on complete and incomplete data.
    print(f"  >> Rows with complete data:   {len(df_complete)} ({len(df_complete)/len(data):.1%})")
    print(f"  >> Rows with incomplete data: {len(df_incomplete)} ({len(df_incomplete)/len(data):.1%})")
    # -----------------------

    # ---- Con los datos incompletos se estima el modelo MAR de datos de perovskita.
    mar_model = model_MAR_pattern(df_incomplete, threshold_pattern)

    # 2. YOU MUST FIT FIRST
    # This creates the 'colname_to_idx' attribute the error is complaining about
    mar_model.fit(df_incomplete) 

    the_models = [
        # --- The Mean Baseline ---
        {'estimator': DummyRegressor(strategy='mean'), 'max_iter': 1, 'name': 'Mean_Baseline'},
        {'estimator': RandomForestRegressor(n_estimators=50, n_jobs=1, random_state=42), 'max_iter': 50,'name': 'MissForest_Classic_50'},
        {'estimator': ExtraTreesRegressor(n_estimators=50, n_jobs=1, random_state=42), 'imputation_order': 'ascending', 'max_iter': 50, 'name': 'MissForest_Extra_50'},
        {'estimator': KNeighborsRegressor(n_neighbors=5), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_KNN_5'},
        {'estimator': BayesianRidge(), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_Bayesian_Asc'},
        {'estimator': Ridge(random_state=42), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_Ridge'}
        ]
    
    # --- MAIN EVALUATION LOOP ---
    all_scores = {config['name']: [] for config in the_models}
    for r in range(R):                    #-- MonteCarlo: we generate enough MAR missing paterns.
        # 1. Shuffle and Split the RAW (unscaled) data
        df_complete = df_complete.sample(frac=1, random_state=r).reset_index(drop=True)
        df_train_true_raw, df_test_true_raw = train_test_split(df_complete, test_size=0.20, random_state=r)
        # 2. Ampute the RAW data (This is where the MAR logic works best!)
        X_train_mar = mar_model.transform(df_train_true_raw)
        df_train_miss_raw = pd.DataFrame(X_train_mar, columns=df_train_true_raw.columns, index=df_train_true_raw.index)
        X_test_mar = mar_model.transform(df_test_true_raw)
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
            nE_dist = normalized_energy_distance(df_test_true, df_test_imputed)
            rel_score = distribution_preservation_score(df_test_true, df_test_imputed, mean_imputed_data)

            # STORE ALL METRICS in a dictionary for this repetition
            iteration_results = {'repetition': int(r),
                        'nrmse': float(normalized_rmse),
                        'nmae': float(normalized_mae),
                        'energy_distance': float(nE_dist),
                        'dist_preservation_score': float(rel_score),
                        'wasserstein' : float(dist_emd_wasserstein)
                        }
            print(iteration_results)

            all_scores[config['name']].append(iteration_results)

        print(f"Repetition {r+1}/{R} complete.")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        with open('all_scores_checkpoint.json', 'w') as f:
        	json.dump(all_scores, f)

    ## --- POST-PROCESSING EXAMPLES ---
    #print("\n--- FINAL RESULTS ---")
    #for model_name, scores in all_scores.items():
    #   print(f"{model_name} -> Mean NRMSE: {np.mean(scores):.4f} | Std: {np.std(scores):.4f}")

    #print("\n" + "="*50)
    #print(" 📊 FINAL MISSFOREST EVALUATION REPORT")
    #print("="*50)

    #file_name = "final_evaluation_results.csv"
    #rganize_results(all_scores, file_name)