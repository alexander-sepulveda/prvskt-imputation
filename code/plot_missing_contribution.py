import sys
#sys.path.insert(0, './data')  # add src to Python path

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import chisquare

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




def plot_missing_contribution(df):
    """
    Creates a bar plot showing the contribution of each column to the total
    number of missing values in the dataset.
    """
    # 1. Count missing values per column
    missing_counts = df.isnull().sum()
    
    # 2. Calculate the total number of missing values in the entire dataframe
    total_missing = missing_counts.sum()
    
    # Handle the case where there are zero missing values to avoid division by zero
    if total_missing == 0:
        print("No missing values found in the dataset.")
        return

    # 3. Calculate the share (%) of total missingness for each column
    missing_share = (missing_counts / total_missing) * 100
    
    # 4. Sort columns by contribution (Descending)
    missing_share = missing_share.sort_values(ascending=False)
    
    # 5. Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(missing_share.index, missing_share.values, color='salmon', edgecolor='darkred')
    
    # 6. Formatting and Labels
    ax.set_ylabel('Share of Total Missing Values (%)', fontsize=12)
    ax.set_xlabel('Columns', fontsize=12)
    ax.set_title('Contribution per Column to Total Dataset Missingness', fontsize=14)
    
    # Rotate x-labels to prevent overlapping
    plt.xticks(rotation=45, ha='right')
    
    # Add text labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        if height > 0: # Only label bars that have missing values
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, 
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('missing_contribution_histogram.png')


def compare_missing_contribution(df1, df2, label1='Incomplete', label2='Raw'):
    """
    Creates a grouped bar plot comparing the contribution of each column to the 
    total number of missing values across two dataframes.
    """
    # 1. Calculate shares for the first dataframe
    m1 = df1.isnull().sum()
    t1 = m1.sum()
    s1 = (m1 / t1 * 100) if t1 > 0 else m1 * 0
    
    # 2. Calculate shares for the second dataframe
    m2 = df2.isnull().sum()
    t2 = m2.sum()
    s2 = (m2 / t2 * 100) if t2 > 0 else m2 * 0
    
    # 3. Combine and align indices
    df_comp = pd.DataFrame({label1: s1, label2: s2}).fillna(0)
    
    # 4. Sort by the first dataframe for consistent visualization
    df_comp = df_comp.sort_values(by=label1, ascending=False)
    
    # 5. Plotting setup
    x = np.arange(len(df_comp.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create grouped bars
    rects1 = ax.bar(x - width/2, df_comp[label1], width, label=label1, color='salmon', edgecolor='darkred')
    rects2 = ax.bar(x + width/2, df_comp[label2], width, label=label2, color='skyblue', edgecolor='navy')
    
    # 6. Formatting
    ax.set_ylabel('Share of Total Missing Values (%)', fontsize=12)
    ax.set_title('Comparison of Missing Value Contribution by Column', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(df_comp.index, rotation=45, ha='right')
    ax.legend()
    
    # Add labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('missing_contribution_comparison.png')



def compare_pmfs_chi2(df, col_empirical_E, col_estimated_O, n_experiments=30):
    #  Performs a Chi-squared Goodness-of-Fit test comparing discrete probability mass functions.
    # 1. Extract the arrays as floats
    expected_E = df[col_empirical_E].values.astype(float)
    observed_O = df[col_estimated_O].values.astype(float)
    
    # 2. Convert percentages to absolute frequencies (counts) based on n_experiments
    expected_counts = (expected_E / 100.0) * n_experiments
    observed_counts = (observed_O / 100.0) * n_experiments

    # 3. Filter out zero-divisions
    # Chi-square formula divides by E, so we must remove rows where Expected (Empirical) is 0
    valid_mask = expected_counts > 0
    exp_valid = expected_counts[valid_mask]
    obs_valid = observed_counts[valid_mask]
    
    # 4. Normalize the observed counts to perfectly match the sum of expected counts
    # (scipy.stats.chisquare strictly requires the sum of O to equal the sum of E)
    obs_valid = obs_valid * (np.sum(exp_valid) / np.sum(obs_valid))

    # 5. Perform the Chi-squared test
    # f_obs corresponds to the estimated data, f_exp corresponds to the empirical reference
    chi2_stat, p_value = chisquare(f_obs=obs_valid, f_exp=exp_valid)
    
    print(f"Number of valid categories: {len(obs_valid)}")
    print(f"Degrees of freedom: {len(obs_valid) - 1}")
    print(f"Chi-square Statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    return chi2_stat, 1-p_value


def compare_missing_contribution_custom(df1, df2, label_mapping=None, label1='Incomplete', label2='Raw'):
    """
    Compares missingness contribution using a dictionary to rename column labels.
    
    Parameters:
    - label_mapping: dict, e.g., {'old_col_name': 'New Descriptive Label'}
    """
    cols_to_drop = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4']
    # Drop columns from df1 and df2
    df1 = df1.drop(columns=cols_to_drop)
    df2 = df2.drop(columns=cols_to_drop)
    # 1. Calculate shares for both dataframes
    s1 = (df1.isnull().sum() / df1.isnull().sum().sum() * 100).fillna(0)
    s2 = (df2.isnull().sum() / df2.isnull().sum().sum() * 100).fillna(0)
    
    # 2. Combine into a comparison DataFrame
    df_comp = pd.DataFrame({label1: s1, label2: s2}).fillna(0)
    
    # 3. Apply the label mapping if provided
    if label_mapping:
        # We rename the index (which contains the column names)
        df_comp.index = [label_mapping.get(col, col) for col in df_comp.index]
    
    # 4. Sort for visual clarity
    df_comp = df_comp.sort_values(by=label1, ascending=False)

    print('***************')
    print(df_comp)
    chi2_stat, p_value = compare_pmfs_chi2(df_comp, col_empirical_E='Empirical missing pattern', col_estimated_O='Estimated missing pattern', n_experiments=30)
    print(chi2_stat)
    print(p_value)
    print('***************')
    
    # 5. Plotting
    x = np.arange(len(df_comp.index))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(x - width/2, df_comp[label1], width, label=label1, color='gray', edgecolor='black')
    ax.bar(x + width/2, df_comp[label2], width, label=label2, color='black', edgecolor='gray')
    
    # Formatting
    ax.set_ylabel('Contribution(%)', fontsize=16)
    ax.set_title('Missing Value Contribution', fontsize=16)
    ax.set_xticks(x)
    #ax.set_xticklabels(df_comp.index, rotation=45, ha='right')
    ax.set_xticklabels(df_comp.index, ha='right', fontsize=16)
    ax.legend(fontsize=16)
    
    plt.tight_layout()
    plt.savefig('missing_contribution.png')



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


# ---- Little's MCAR test. --------
from pyampute.exploration.mcar_statistical_tests import MCARTest

mt = MCARTest(method="little")
p_value = mt.little_mcar_test(df_incomplete)
print(f"Little's MCAR Test p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Conclusion: Reject MCAR (Missingness is likely MAR or MNAR)")
else:
    print("Conclusion: Fail to reject MCAR (Data may be missing completely at random)")

# --- histogram of missing values.
#plot_missing_contribution(df_incomplete)


# ---
threshold_pattern = 0.005
mar_model = model_MAR_pattern(df_incomplete, threshold_pattern)
mar_model.fit(df_incomplete)
X_train_mar = mar_model.transform(df_complete)
df_miss_raw = pd.DataFrame(X_train_mar, columns=df_complete.columns, index=df_complete.index)
#plot_missing_contribution(df_miss_raw)

prvskt_labels = {
    'LLE_1': 'L_1',
    'LLE_2': 'L_2',
    'LLE_3': 'L_3',
    'LLE_4': 'L_4',
    'DMF_DMSO_ratio': r'$\chi_{\text{sol.}}$',
    'first_Prvskt_annealing_temperature': r'$T_1$',
    'first_Prvskt_thermal_annealing_time': r'$t_1$',
    'Perovskite_annealing_thermal_exposure' : r'$TB$',
    'Perovskite_band_gap' : r'$E_g$',
    'Perovskite_thickness' : r'$\psi$',
    'Cell_area_measured' : r'$\mathcal{A}$',
    'JV_default_Voc' : r'$V_{oc}$',
    'JV_default_Jsc' : r'$J_{sc}$',
    'JV_default_FF' : r'$FF$',
    'JV_default_PCE' : r'$PCE$'
}

compare_missing_contribution_custom(df_incomplete, df_miss_raw, prvskt_labels, label1='Empirical missing pattern', label2='Estimated missing pattern')
plt.show()