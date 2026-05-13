# show the plots.
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd


def plot_models_with_ci(data_dict, selected_models, title="Model Performance (with 95% CI)"):
    """
    Plots a bar chart for selected models from a dictionary, showing Confidence Intervals.
    
    Parameters:
    - data_dict: Dictionary containing model metrics.
    - selected_models: List of model names to plot.
    - title: Title of the plot.
    """
    # 1. Filter out any models that don't exist in the dictionary to prevent errors
    valid_models = [m for m in selected_models if m in data_dict]
    
    if not valid_models:
        print("Error: None of the selected models were found in the dictionary.")
        return

    # 2. Extract the metrics safely
    means = [data_dict[m].get('Mean_NRMSE', 0) for m in valid_models]
    ci_lowers = [data_dict[m].get('CI_Lower', 0) for m in valid_models]
    ci_uppers = [data_dict[m].get('CI_Upper', 0) for m in valid_models]

    print(ci_lowers)
    print(ci_uppers)
    
    # 3. Calculate the relative error lengths for matplotlib
    # Matplotlib needs the *distance* from the mean, not the absolute CI values
    lower_errors = [mean - ci_low for mean, ci_low in zip(means, ci_lowers)]
    upper_errors = [ci_up - mean for mean, ci_up in zip(means, ci_uppers)]
    
    # Pack them into a 2D list: [ [lower_errors], [upper_errors] ]
    asymmetric_errors = [lower_errors, upper_errors]
    
    # 4. Plotting setup
    x = np.arange(len(valid_models))
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw the bars with the asymmetric CI error bars
    ax.bar(x, means, yerr=asymmetric_errors, capsize=6, color='teal', alpha=0.8, edgecolor='black')
    
    # 5. Formatting
    ax.set_ylabel('Mean NRMSE', fontweight='bold', fontsize=12)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_models, rotation=45, ha='right', fontsize=11)
    
    # Add a clean grid behind the bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def plot_imputation_metrics(json_filename):
    name_mapping = {
        'Mean_Baseline' : 'Mean Imputation',
        'MissForest_Classic_50': 'MissForest (Random Forest) [50 trees]',
        'MissForest_Extra_50' : 'MissForest with Extra Trees ensemble [50 trees]',
        'MICE_KNN_5': 'MICE with K-Nearest Neighbors [K=5]',
        'MICE_Bayesian_Asc' : 'MICE with bayesian linear regression',
        'MICE_Ridge' : 'MICE with Ridge Regression'
        }

    # 1. Load and Prepare Data
    with open(json_filename, 'r') as f:
        all_scores = json.load(f)
        
    rows = []
    for raw_name, iterations in all_scores.items():
        clean_name = name_mapping.get(raw_name, raw_name)
        for iteration in iterations:
            row = iteration.copy()
            row['Model'] = clean_name
            rows.append(row)
    df = pd.DataFrame(rows)
    df['energy_distance'] = 72.8*df['energy_distance'] # antes grafiacaba la normalizada 
    df['energy_distance'] = np.sqrt(df['energy_distance']/2)
    #respecto a la máxima posible 2*np.sqrt(N). Ahora, prefiero graficar la natural.

    print(df)
    
    metrics_config = {
        'nrmse': 'a) Normalized RMSE',
        'nmae': 'b) Normalized MAE',
        'energy_distance': 'c) Energy Distance',
        'dist_preservation_score': 'd) Distribution Preservation (Higher is Better)',
        'wasserstein': 'e) Wasserstein Distance'
    }
    metrics_to_plot = {k: v for k, v in metrics_config.items() if k in df.columns}
    
    # 2. Set up Plotting Grid (3 rows, 2 columns)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes_flat = axes.flatten()
    
    # 3. Generate Metric Plots
    for i, (metric, title) in enumerate(metrics_to_plot.items()):
        ax = axes_flat[i]
        sns.barplot(
            data=df, 
            x='Model', 
            y=metric, 
            hue='Model', 
            legend=True, # We enable it temporarily to "catch" it for the global legend
            ax=ax,
            capsize=0.1,
            errorbar=('ci', 95),
            edgecolor='black',
            palette='Set1'
        )
        
        # Cleanup individual plot
        ax.set_title(title, fontweight='normal', fontsize=12)
        ax.set_ylabel('')
        #ax.set_xlabel('')
        ax.set_xlabel(title, fontweight='normal', fontsize=12, labelpad=10)
        ax.set_title('')
        #ax.tick_params(axis='x', labelbottom=False) # Hide x-labels since legend handles it
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        
        # Grab the legend handles/labels from the first plot then remove it from the axis
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    # 4. Create the Dedicated Legend Plot in the last slot
    # Use the 6th slot (index 5) for the legend
    legend_ax = axes_flat[5] 
    legend_ax.axis('off') # Hide the grid/axes
    
    legend_ax.legend(
        handles, 
        labels, 
        loc='center', 
        title="Imputation Methods", 
        title_fontsize='12',
        fontsize='12',
        frameon=True,
        shadow=True,
        facecolor='#f0f0f0'
    )

    # 5. Final Cleanup
    # If there are other empty slots (besides the legend slot), remove them
    for j in range(len(metrics_to_plot), 5): 
        fig.delaxes(axes_flat[j])
        
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1) # Make sure legend has room
    plt.savefig('imputation_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# --- To use the function: ---
plot_imputation_metrics('all_scores_checkpoint.json')