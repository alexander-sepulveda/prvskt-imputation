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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
import umap

from MAR import preprocessing_data_prvskt
from MAR import model_MAR_pattern

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
    df_true['Source'] = 'Original (Reference)'
    df_imputed['Source'] = 'Imputed (MissForest)'

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
        data=plot_df[plot_df['Source'] == 'Original (Reference)'],
        x='UMAP 1', y='UMAP 2',
        color='black', alpha=0.9, label='Original (Reference)',
        s=20, edgecolor=None
    )

    # Plot Imputed data on top (distinct color)
    sns.scatterplot(
        data=plot_df[plot_df['Source'] == 'Imputed (MissForest)'],
        x='UMAP 1', y='UMAP 2',
        color='#1f77b4', alpha=0.6, label='Imputed (MissForest)',
        s=25, marker='o'
    )

    plt.title(title, fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.show()



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
	# ----- hyper-parameters ---
	# the variables we include in the analysis.
    var_names = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4', 'DMF_DMSO_ratio', 'first_Prvskt_annealing_temperature',
             'first_Prvskt_thermal_annealing_time', 'Perovskite_annealing_thermal_exposure', 'Perovskite_band_gap', 'Perovskite_thickness',
             'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    # outliers removal before splitting on complete and incomplete data.
    cols_to_fix = ['first_Prvskt_thermal_annealing_time', 'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    skewed_cols = ['first_Prvskt_thermal_annealing_time']    # variables with long-tails.
    threshold_pattern = 0.005                                 # threshold to discard those 1% rare patterns.
    R = 1                                                   # Monte Carlo Repetitions
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
    the_model = {'estimator': ExtraTreesRegressor(n_estimators=50, n_jobs=1, random_state=42), 'imputation_order': 'ascending', 'max_iter': 50, 'name': 'MissForest_Extra_50'}
    
    # --- MAIN EVALUATION LOOP ---
    #all_scores = {config['name']: [] for config in the_models}
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
        imputer = IterativeImputer(estimator=the_model['estimator'], max_iter=the_model['max_iter'], random_state=42)
        imputer = imputer.fit(df_train_miss)
        test_imputed  = imputer.transform(df_test_miss)
        df_test_imputed = pd.DataFrame(test_imputed, columns=df_test_true.columns, index=df_test_true.index)
        array_true = np.asarray(df_test_true)
        array_imp = np.asarray(df_test_imputed)

        plot_umap_validation(df_test_true, df_test_imputed)


        print(f"Repetition {r+1}/{R} complete.")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')