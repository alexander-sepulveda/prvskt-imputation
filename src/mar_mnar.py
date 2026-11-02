import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyampute import MultivariateAmputation

import matplotlib.pyplot as plt
import seaborn as sns

import copy
from pyampute.ampute import MultivariateAmputation


def remove_outliers_iqr(df, columns, factor=2.5):
    df_clean = df.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        stat_lower = Q1 - (factor * IQR)
        lower_bound = max(0, stat_lower) 
        upper_bound = Q3 + (factor * IQR)
        
        # Now 0 values will be kept because 0 >= 0 is True. NaNs are kept too, otherwise
        # every incomplete row would be dropped and the missingness patterns lost.
        mask = df_clean[col].isna() | ((df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound))
        df_clean = df_clean[mask]

    return df_clean


def preprocessing_data_prvskt(data, var_names):
    # Separate complete data and incomplete data; but before, it eliminates observations with no material reported.

    # Eliminate rows where the material is 'null'
    first_four = var_names[:4]                       # represents material
    data = data.dropna(subset=first_four, how='all')

    # remove double layer perovskites. Our scope is single layer perovskites.
    data = data[~data['Perovskite_thickness'].str.contains('|', regex=False, na=False)]

    # Separates "Trash Strings" from genuine NaNs
    was_originally_nan = data['Perovskite_thickness'].isna()   # Check which rows are ALREADY NaN
    converted_values = pd.to_numeric(data['Perovskite_thickness'], errors='coerce')   # Try to convert to numeric
    is_trash_symbol = converted_values.isna() & ~was_originally_nan  # Identify the "Trash" rows
    initial_count = len(data)
    data = data[~is_trash_symbol].copy()                         # Filter them out

    data['Perovskite_thickness'] = pd.to_numeric(data['Perovskite_thickness'], errors='coerce')   # cleaned numeric values
    data = data.dropna(how='all')               # Eliminate rows where all values in the row are 'null'

    data = data.reset_index(drop=True)
    is_incomplete = data.isnull().any(axis=1)   # True when 'this row has at least one NaN'
    df_incomplete = data[is_incomplete].copy()  # rows with at least one missing value
    df_complete   = data[~is_incomplete].copy()   # rows with NO missing values


    return df_complete, df_incomplete


def model_MAR_pattern(df_incomplete, threshold, real_prop, flag_plot=False):
    # having incomplete data 'df_incomplete' it estimates the model describing the missingness MAR model
    # representing that missingness pattern.
    #
    # df_incomplete : incomplete data.
    # threshold     : used to discard those less frequent paterns
    # ma : returns MAR model. 

    final_proportions = get_proportion_patterns(df_incomplete, threshold)
    df_freq_incomplete = get_most_frequent(df_incomplete, final_proportions)
    if flag_plot:  # Conditionally execute the plotting function
        missing_vars_patterns_2(final_proportions)
    final_pyampute_list = pyampute_input_patterns(df_freq_incomplete, final_proportions)

    ma = MultivariateAmputation(patterns=final_pyampute_list, prop=real_prop)
    
    return ma

def get_proportion_patterns(df_incomplete, threshold):
    # determine the number of unique missing patterns.
    data = df_incomplete
    mask = data.notnull().astype(int)                                 # Create a binary mask (1 = observed, 0 = missing).
    pattern_stats = mask.value_counts(normalize=True)                 # patterns and their probabilities
    frequent_patterns = pattern_stats[pattern_stats >= threshold]     # Filter patterns at least X% (threshold*100) of the data
    incomplete_patterns = frequent_patterns.drop(labels=tuple([1]*len(data.columns)), errors='ignore')
    prop_acumulada = incomplete_patterns.sum()
    print('*************')
    print('proporción acumulada:')
    print(incomplete_patterns.sum())
    final_proportions   = incomplete_patterns / incomplete_patterns.sum()

    return final_proportions

def get_most_frequent(df_incomplete, final_proportions):
    # It takes those rows whose missing patterns are among the most frequent.

    # binary mask for the entire incomplete dataframe. 1 = observed, 0 = missing
    mask_all = df_incomplete.notnull().astype(int)
    row_patterns = mask_all.apply(tuple, axis=1)        # Series where each element is a tuple like (1, 0, 1, 0...)
    is_frequent = row_patterns.isin(final_proportions.index) # Is this row's pattern in our 'final_proportions'?
    df_freq_incomplete = df_incomplete[is_frequent].copy()   # Filter the dataframe

    df_freq_incomplete = df_freq_incomplete.reset_index(drop=True)    # Clean up the index

    return df_freq_incomplete

def pyampute_input_patterns(df_freq_incomplete, final_proportions):
    final_pyampute_list = []
    # Loop through each of your frequent patterns
    for mask_tuple, proportion in final_proportions.items():
        # Identify which variables are missing in this pattern (where mask is 0)
        incomplete_vars = [i for i, val in enumerate(mask_tuple) if val == 0]
        # Estimate the weights based on REAL data
        weights = estimate_mar_weights(df_freq_incomplete, mask_tuple)
        # We zero out the weights for the variables that are missing in this pattern.
        for idx in incomplete_vars:
            weights[idx] = 0.0

        # Build the dictionary for MultivariateAmputer
        final_pyampute_list.append({
            "incomplete_vars": incomplete_vars,
            "weights": weights,
            "freq": proportion,
            "mechanism": "MAR",
            "score_to_probability_func": "SIGMOID-RIGHT"
            })

    return final_pyampute_list

def estimate_mar_weights(df, target_pattern_mask):
    # df: incomplete data.
    # target_pattern_mask: The specific tuple (1, 0, 1...) we are looking for

    # 1. Create the binary target: 1 if the row matches THIS pattern, 0 if it matches ANOTHER pattern
    current_masks = df.notnull().astype(int)
    y = (current_masks == target_pattern_mask).all(axis=1).astype(int)

    if len(np.unique(y)) == 1:
        print(f"Warning: Pattern {target_pattern_mask} has only 1 class. Defaulting to 0 weights.")
        return np.zeros(df.shape[1])

    X = df.copy()

    for col in X.columns:
        col_mean = X[col].mean()
        fill_value = 0 if pd.isna(col_mean) else col_mean  # If the column is entirely NaN, col_mean is NaN. Fallback to 0.
        X[col] = X[col].fillna(fill_value)

    # 4. Scale and Fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # We use a standard penalty to keep weights realistic
    model = LogisticRegression(solver='lbfgs', C=1.0, max_iter=1000)
    model.fit(X_scaled, y)

    return model.coef_[0]

def missing_vars_patterns_2(final_proportions):
    var_names = ['L_1', 'L_2', 'L_3', 'L_4', 'DMF:DMSO solvents ratio', 'first annea. temperature',
                 'first annea. time', 'thermal exposure', 'band gap', 'thickness',
                 'cell area measured', 'Voc', 'Jsc', 'FF', 'PCE']

    patterns_list = []
    for mask_tuple, proportion in final_proportions.items():
        # Identify which variables (column indices) are missing in this pattern
        # In the mask, 0 = missing
        incomplete_vars = [i for i, val in enumerate(mask_tuple) if val == 0]
        pattern_dict = {
                "incomplete_vars": incomplete_vars,
                "prop": proportion,
                "mechanism": "MAR",
                }
        patterns_list.append(pattern_dict)
    
    # Get the unique integer indices of variables that are missing across all patterns
    missing_indices = sorted(list(set([v for p in patterns_list for v in p['incomplete_vars']])))
    
    # Map these integer indices to their actual string names using your var_names list
    y_labels = [var_names[i] for i in missing_indices]
    
    # Create the matrix (Patterns x Missing Variables)
    matrix = np.zeros((len(patterns_list), len(missing_indices)))
    for i, p in enumerate(patterns_list):
        for v in p['incomplete_vars']:
            # map the original column index 'v' to its position in the missing_indices list
            matrix[i, missing_indices.index(v)] = 1
            
    matrix_t = matrix.T
    
    # Shared X-axis variables
    props = [p['prop'] * 100 for p in patterns_list]
    x_pos = np.arange(len(patterns_list))
    pattern_labels = [f"{i+1}" for i in range(len(patterns_list))]

    # ---------------------------------------------------------
    # Figure 1: Bar Chart (Pattern Prevalence)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    
    ax1.bar(x_pos, props, color='black', width=0.8) 
    ax1.set_ylabel("Frequency (%)",fontsize=14)
    ax1.set_xlabel("Patterns", fontsize=14)
    ax1.set_title("Pattern Prevalence", fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(pattern_labels, fontsize=14)
    
    plt.tight_layout()

    # ---------------------------------------------------------
    # Figure 2: Heatmap (Missing Variables Matrix)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(matrix_t, annot=False, cbar=False, cmap="Greys", linewidths=0.5,
                xticklabels=pattern_labels, 
                yticklabels=y_labels, square=True, 
                ax=ax2)

    # Make Variable Labels Horizontal
    ax2.set_yticklabels(y_labels, rotation=0, va='center', fontsize=14)
    ax2.set_xlabel("Patterns", fontsize=14)
    ax2.set_title("Missing Variables per Pattern", fontsize=14)
    #ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pattern_labels, fontsize=14)
    
    plt.tight_layout()

    # Show both figures
    plt.show()


def ampute_mar_mnar(df_complete, ma_mar, gamma, seed=None):
    """
    Ampute a complete DataFrame with a MAR+MNAR mechanism using pyampute.
    df_complete : pandas DataFrame, complete (no NaN), numeric.
    ma_mar      : your fitted pyampute MultivariateAmputation (MAR).
    gamma       : weight given to the variable's own value.
                    0  -> MAR (same as your ma_mar)
                   >0  -> high values are more likely to go missing (MNAR)
                   <0  -> low values are more likely to go missing  (MNAR)
                  Same scale as your fitted MAR weights.
    seed        : random seed (use the same seed for every gamma).

    Returns
    -------
    df_missing  : copy of df_complete with NaN in the amputed cells.
    """

    new_patterns = copy.deepcopy(ma_mar.patterns)

    # For each pattern, give its incomplete variables the weight gamma.
    # making gamma <> 0 makes missingness depend on that value too: MAR + MNAR.
    for pattern in new_patterns:
        weights = np.array(pattern["weights"], dtype=float)
        for j in pattern["incomplete_vars"]:
            weights[j] = gamma
        pattern["weights"] = weights

        if gamma != 0:
            pattern["mechanism"] = "MAR+MNAR"

    # Build a new pyampute amputer with the new patterns and the
    # same proportion of missing rows as your MAR model.
    ma_new = MultivariateAmputation(prop=ma_mar.prop,
                                    patterns=new_patterns,
                                    seed=seed)

    # Let pyampute do the amputation.
    X_missing = ma_new.fit_transform(df_complete.to_numpy(dtype=float))

    # Return it as a DataFrame with the original columns and index.
    df_missing = pd.DataFrame(X_missing,
                              columns=df_complete.columns,
                              index=df_complete.index)
    return df_missing