import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pyampute import MultivariateAmputation

import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing_data_prvskt(data, var_names):
    # apply some basic filters and separate complete data and incomplete data.
    
    first_four = var_names[:4]  # ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4']
    data = data.dropna(subset=first_four, how='all') # Eliminate rows where the material ( first 4 columns) are 'null'
    
    #data = data[~data['Perovskite_thickness'].str.contains('\|', na=False)]   # remove double layer perovskites.
    data = data[~data['Perovskite_thickness'].str.contains('|', regex=False, na=False)]
    was_originally_nan = data['Perovskite_thickness'].isna()   # Check which rows are ALREADY NaN
    converted_values = pd.to_numeric(data['Perovskite_thickness'], errors='coerce')   # Try to convert to numeric
    is_trash_symbol = converted_values.isna() & ~was_originally_nan  # Identify the "Trash" rows
    initial_count = len(data)
    data = data[~is_trash_symbol].copy()                         # Filter them out
    data['Perovskite_thickness'] = pd.to_numeric(data['Perovskite_thickness'], errors='coerce')   # cleaned numeric values

    data = data.dropna(how='all')                # Eliminate rows where all values in the row are 'null'
    print('Know we have this number of rows: ')
    print(len(data))

    data = data.reset_index(drop=True)
    is_incomplete = data.isnull().any(axis=1)   # True when 'this row has at least one NaN'
    df_incomplete = data[is_incomplete].copy()  # rows with at least one missing value
    df_complete   = data[~is_incomplete].copy()   # rows with NO missing values
    # 3. Quick Audit
    #print(f"Total rows in original data: {len(data)}")
    #print(f"Rows with complete data:     {len(df_complete)} ({len(df_complete)/len(data):.1%})")
    #print(f"Rows with incomplete data:   {len(df_incomplete)} ({len(df_incomplete)/len(data):.1%})")

    return df_complete, df_incomplete



def get_proportion_patterns(df_incomplete, threshold):
    # determine the number of unique missing patterns.
    data = df_incomplete
    mask = data.notnull().astype(int)                                 # Create a binary mask (1 = observed, 0 = missing).
    pattern_stats = mask.value_counts(normalize=True)                 # patterns and their probabilities
    frequent_patterns = pattern_stats[pattern_stats >= threshold]     # Filter patterns at least X% of the data
    incomplete_patterns = frequent_patterns.drop(labels=tuple([1]*len(data.columns)), errors='ignore')
    final_proportions   = incomplete_patterns / incomplete_patterns.sum()

    return final_proportions

def get_most_frequent(df_incomplete, final_proportions):
    # It takes those rows whose missing patters are among the most frequent.

    # binary mask for the entire incomplete dataframe. 1 = observed, 0 = missing
    mask_all = df_incomplete.notnull().astype(int)
    row_patterns = mask_all.apply(tuple, axis=1)        # Series where each element is a tuple like (1, 0, 1, 0...)
    is_frequent = row_patterns.isin(final_proportions.index) # Is this row's pattern in our 'final_proportions'?
    df_freq_incomplete = df_incomplete[is_frequent].copy()   # Filter the dataframe

    df_freq_incomplete = df_freq_incomplete.reset_index(drop=True)    # Clean up the index
    # --- Verification ---
    print(f"Initial incomplete rows: {len(df_incomplete)}")
    print(f"Frequent incomplete rows: {len(df_freq_incomplete)}")
    print(f"Rows removed (rare patterns): {len(df_incomplete) - len(df_freq_incomplete)}")

    return df_freq_incomplete


def missing_vars_patterns(final_proportions):
    var_names = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4', 'DMF:DMSO solvents ratio', 'first annealing temperature',
             'first annealing time', 'annealing thermal exposure', 'perovskite band gap', 'thickness',
             'measured cell area', 'Voc', 'Jsc', 'FF', 'PCE']

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

    print(f"Extracted {len(patterns_list)} major missingness patterns.")
    
    # 1. Prepare Matrix
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

    # 2. Plotting
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 4]})

    # --- Top: Bar Chart ---
    props = [p['prop'] * 100 for p in patterns_list]
    x_pos = np.arange(len(patterns_list))
    
    # We use 0.5 offset because Heatmap cells start at 0.5 center
    ax2.bar(x_pos + 0.5, props, color='black', width=0.8) 
    ax2.set_ylabel("Frequency (%)")
    ax2.set_title("Pattern Prevalence")
    
    # Remove extra labels from top plot
    ax2.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # --- Bottom: Heatmap ---
    sns.heatmap(matrix_t, annot=False, cbar=False, cmap="Greys", linewidths=0.5,
                xticklabels=[f"P{i+1}" for i in range(len(patterns_list))], 
                yticklabels=y_labels, square=True, # Use the actual variable names!
                ax=ax1)

    # 1. Align the X-axis perfectly
    ax1.set_xlim(0, len(patterns_list))
    ax2.set_xlim(0, len(patterns_list))

    # 2. Make Variable Labels Horizontal
    # rotation=0 makes them horizontal; va='center' aligns them to the row
    ax1.set_yticklabels(y_labels, rotation=0, va='center')

    ax1.set_xlabel("Patterns")
    #ax1.set_ylabel("Variables")

    # Tighten the space between plots
    plt.subplots_adjust(hspace=0.08)
    plt.show()


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

    print(f"Extracted {len(patterns_list)} major missingness patterns.")
    
    # 1. Prepare Matrix
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
    pattern_labels = [f"P{i+1}" for i in range(len(patterns_list))]

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


def estimate_mar_weights(df, target_pattern_mask):
    """
    df: df_frequent_incomplete (the real messy data)
    target_pattern_mask: The specific tuple (1, 0, 1...) we are looking for
    """
    # 1. Create the binary target: 1 if the row matches THIS pattern, 0 if it matches ANOTHER frequent pattern
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


def pyampute_input_patterns(df_freq_incomplete, final_proportions):
    final_pyampute_list = []
    # Loop through each of your frequent patterns
    for mask_tuple, proportion in final_proportions.items():
        # Identify which variables are missing in this pattern (where mask is 0)
        incomplete_vars = [i for i, val in enumerate(mask_tuple) if val == 0]
        # Estimate the weights based on REAL data
        weights = estimate_mar_weights(df_freq_incomplete, mask_tuple)
        print('--------------')
        print(weights)
        # --- IMPORTANT: FORCE PURE MAR ---
        # A variable cannot cause itself to be missing.
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
    print(f"Successfully prepared {len(final_pyampute_list)} patterns for pyampute.")

    return final_pyampute_list

def model_MAR_pattern(df_incomplete, threshold):
    # having incomplete data 'df_incomplete' it estimates the model describing the missingness MAR model
    # representing that missingness pattern.
    # df_incomplete : incomplete data.
    # threshold     : used to discard those less frequent paterns

    final_proportions = get_proportion_patterns(df_incomplete, threshold)
    print(final_proportions)
    print('*****************************')
    df_freq_incomplete = get_most_frequent(df_incomplete, final_proportions)
    #--missing_vars_patterns_2(final_proportions)
    final_pyampute_list = pyampute_input_patterns(df_freq_incomplete, final_proportions)

    ma = MultivariateAmputation(patterns=final_pyampute_list)
    
    return ma


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




if __name__ == "__main__":
    # the variables we include in the analysis.
    var_names = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4', 'DMF_DMSO_ratio', 'first_Prvskt_annealing_temperature',
                 'first_Prvskt_thermal_annealing_time', 'Perovskite_annealing_thermal_exposure', 'Perovskite_band_gap', 'Perovskite_thickness',
                 'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    # outliers removal before splitting on complete and incomplete data.
    cols_to_fix = ['first_Prvskt_thermal_annealing_time', 'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    skewed_cols = ['first_Prvskt_thermal_annealing_time']    # variables with long-tails.

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

    # ---------
    
    #threshold = 0.00001    # threshold to discard those rare patterns.
    threshold = 0.005       # we use 0.5% in the CMC article.
    mar_model = model_MAR_pattern(df_incomplete, threshold)

    mar_model.fit(df_incomplete)
    #X_mar_simulated_array = mar_model.transform(df_complete)
    # Convert to DataFrame
    #df_mar_simulated = pd.DataFrame(X_mar_simulated_array, columns=df_complete.columns)
    #print("Simulation complete!")
    #print(df_mar_simulated.head())

    #nan_percentage = df_mar_simulated.isna().mean() * 100
    #print(nan_percentage)