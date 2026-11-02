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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor

from hyperimpute.plugins.imputers import Imputers
from pyampute.exploration.mcar_statistical_tests import MCARTest

from mar_mnar import remove_outliers_iqr, preprocessing_data_prvskt
from mar_mnar import model_MAR_pattern, ampute_mar_mnar
from performance_metrics import calculate_mean_nrmse, calculate_mean_nmae, emd_wasserstein_pot
from performance_metrics import normalized_energy_distance, distribution_preservation_score
from performance_metrics import organize_results, cmp_missing_vars

import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def plot_umap_validation(df_true, df_imputed, title="UMAP Projection: True vs Imputed Data"):
    """
    Standardizes data, fits a shared UMAP manifold, and plots the overlap.
    """
    # 1. Add labels to distinguish the datasets
    df_true = df_true.copy()
    df_imputed = df_imputed.copy()
    df_true['Source'] = 'Original'
    df_imputed['Source'] = 'Imputed'

    # 2. Combine datasets for a shared manifold
    combined_df = pd.concat([df_true, df_imputed], axis=0).reset_index(drop=True)
    
    # Separate features from the label
    features = combined_df.drop(columns=['Source'])
    labels = combined_df['Source']

    # 3. Scaling (CRITICAL for UMAP)
    # We use StandardScaler to ensure all features contribute equally
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    # 4. Fit UMAP
    # We set random_state=42 for 100% reproducibility (Reviewer-proof)
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        random_state=42,
        low_memory=False
    )
    
    embedding = reducer.fit_transform(scaled_data)

    # Visualization
    plt.figure(figsize=(10, 7), dpi=120)
    plot_df = pd.DataFrame(embedding, columns=['UMAP 1', 'UMAP 2'])
    plot_df['Source'] = labels

    # Plot Original data first (bottom layer, lower alpha)
    sns.scatterplot(
        data=plot_df[plot_df['Source'] == 'Original'],
        x='UMAP 1', y='UMAP 2',
        color='red', alpha=0.9, label='Original',
        s=35, edgecolor=None
    )

    # Plot Imputed data on top (distinct color)
    sns.scatterplot(
        data=plot_df[plot_df['Source'] == 'Imputed'],
        x='UMAP 1', y='UMAP 2',
        color='black', alpha=0.8, label='Imputed (MICE with KNN)',
        s=15, marker='o'
    )

    plt.title(title, fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.show()






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
    R = 1                                                   # number of Monte Carlo Repetitions
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
    cmp_missing_vars(df_incomplete, df_miss_raw, label1='Empirical missing pattern', label2='Estimated missing pattern')
    #import sys
    #sys.exit("Error message")
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------- which methods --------------------------------------------------------------
    if missingness_type == 'MAR':
        gamma_values = [0]
        the_models = [
        # -----------
        {'estimator': KNeighborsRegressor(n_neighbors=5), 'max_iter': 50, 'imputation_order': 'ascending', 'name': 'MICE_KNN_5'},
        ]

    # models for MNAR evaluation in respect the level of missingnes 'gamma'.
    elif missingness_type == 'MNAR':
        print("We test U-map only for the MAR missingness, not for MNAR.")

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
                imputer = IterativeImputer(estimator=config['estimator'], max_iter=config['max_iter'], random_state=42)
                imputer = imputer.fit(df_train_miss)
                test_imputed  = imputer.transform(df_test_miss)
                df_test_imputed = pd.DataFrame(test_imputed, columns=df_test_true.columns, index=df_test_true.index)
                if config['name']== 'Mean_Baseline':
                    mean_imputed_data = df_test_imputed.copy()
                # calculate metrics:
                #normalized_rmse = calculate_mean_nrmse(df_test_true, df_test_imputed, df_test_miss, method='std', verbose=True)
                #normalized_mae  = calculate_mean_nmae(df_test_true, df_test_imputed, df_test_miss, method='std', verbose=False)
                # Force both to be numpy arrays
                array_true = np.asarray(df_test_true)
                array_imp = np.asarray(df_test_imputed)
                plot_umap_validation(df_test_true, df_test_imputed)
