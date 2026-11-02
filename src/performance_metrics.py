import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import ot
import dcor
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2_contingency


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
        The scale is computed on the full df_true column, not only the missing cells.
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
            # We use the full true column, not only the missing subset: under MNAR the
            # missing cells come from a narrow tail and their spread would inflate the score.
            col_true = df_true[col].to_numpy()
            if method == 'std':
                scale = np.std(col_true)
            elif method == 'min_max':
                scale = np.ptp(col_true)
            else:
                scale = 1.0  # Returns raw RMSE

            # 5. Skip constant columns: a raw RMSE would mix units with the normalized scores.
            if scale > 1e-9:
                column_nrmses[col] = rmse / scale
                
    # Verbose output for debugging
    #if verbose and column_nrmses:
    #    print("\n--- Individual Column NRMSE ---")
    #    for c, score in column_nrmses.items():
    #        print(f"{c:20}: {score:.6f}")
            
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
            
            # Using the full true column for normalization (see calculate_mean_nrmse)
            col_true = df_true[col].to_numpy()
            if method == 'std':
                scale = np.std(col_true)
            elif method == 'min_max':
                scale = np.ptp(col_true)
            else:
                scale = 1.0  # Returns raw MAE

            # 5. Skip constant columns to avoid mixing raw MAE with normalized scores
            if scale > 1e-9:
                column_nmaes[col] = mae / scale
                
    #if verbose and column_nmaes:
    #    print("\n--- Individual Column NMAE ---")
    #    for c, score in column_nmaes.items():
    #        print(f"{c:20}: {score:.6f}")
            
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


def nor_energy_distance(true_scaled, imp_scaled):
    # Calculate raw energy distance. It assumes min_max scaling has been applied.
    raw_ed = dcor.energy_distance(true_scaled, imp_scaled)
    sqrt_raw_ed = np.sqrt(raw_ed)
    N = true_scaled.shape[1]
    max_possible_d2 = 2 * np.sqrt(N)
    max_possible_ed = np.sqrt(max_possible_d2)
    E_dist = sqrt_raw_ed / max_possible_ed  #

    return E_dist

def distribution_preservation_score(true_scaled, imp_model_scaled, imp_mean_scaled):
    baseline_dist = dcor.energy_distance(true_scaled, imp_mean_scaled)    # "dumb" mean baseline distance
    #print('>>>> baseline distance')
    #print(baseline_dist)
    model_dist = dcor.energy_distance(true_scaled, imp_model_scaled)
    #print('>>>> model distance')
    #print(model_dist)
    score = 1 - (model_dist / baseline_dist)

    return score