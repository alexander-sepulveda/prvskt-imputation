# Plot imputation benchmark results.
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


NAME_MAPPING = {
    'Mean_Baseline': 'Mean Imputation',
    'Median': 'Median Imputation',
    'MissForest_Classic_50': 'MissForest (Random Forest) [50 trees]',
    'MissForest_Extra_50': 'MissForest with Extra Trees ensemble [50 trees]',
    'MICE_KNN_5': 'MICE with K-Nearest Neighbors [K=5]',
    'MICE_Bayesian_Asc': 'MICE with Bayesian linear regression',
    'MICE_Ridge': 'MICE with Ridge Regression',
    'MIWAE': 'MIWAE',
    'KNN_5': 'K-Nearest Neighbors [K=5]',
}

# Models hidden in specific panels (raw names). The DPS is expected to be positive
# (better than mean imputation), so median imputation is not shown there.
EXCLUDE_FROM_METRIC = {
    'dist_preservation_score': ['Median'],
}

METRICS_CONFIG = {
    'nrmse': 'a) Normalized RMSE',
    'nmae': 'b) Normalized MAE',
    'energy_distance': 'c) Energy Distance',
    'dist_preservation_score': 'd) Distribution Preservation (Higher is Better)',
    'wasserstein': 'e) Wasserstein Distance',
}


def load_scores(json_filename):
    """
    Load results into a long DataFrame with one row per (repetition, model, gamma).

    Accepts both formats:
      - new: a list of records  [{"repetition":..., "model":..., "gamma":..., "nrmse":...}, ...]
      - old: a dict per model   {"Mean_Baseline": [{...}, {...}], ...}
    """
    with open(json_filename, 'r') as f:
        all_scores = json.load(f)

    if isinstance(all_scores, list):                      # new format
        df = pd.DataFrame(all_scores)
    else:                                                 # old format
        rows = []
        for raw_name, iterations in all_scores.items():
            for iteration in iterations:
                row = iteration.copy()
                row['model'] = raw_name
                rows.append(row)
        df = pd.DataFrame(rows)

    if 'gamma' not in df.columns:                         # old files: MAR only
        df['gamma'] = 0.0

    # Keep the order of NAME_MAPPING (unknown models go at the end)
    order = [m for m in NAME_MAPPING if m in df['model'].unique()]
    order += [m for m in df['model'].unique() if m not in order]
    df['Model'] = pd.Categorical(df['model'].map(lambda m: NAME_MAPPING.get(m, m)),
                                 categories=[NAME_MAPPING.get(m, m) for m in order],
                                 ordered=True)

    # al calcularlo en performance_metrics cometí el error de ponerle una raiz cuadrada adicional. Corrijo haciendo esto:
    #df['energy_distance'] = 1.668*df['energy_distance']
    return df


def plot_imputation_metrics(json_filename, gamma=0.0, models=None,
                            output='imputation_results.png'):
    """
    Bar plots (mean ± 95% CI over repetitions) of every metric, for one gamma.

    gamma  : which mechanism to plot (0.0 = MAR). Ignored if the file has one gamma only.
    models : optional list of raw model names to include, e.g. ['Mean_Baseline', 'MICE_KNN_5'].
    """
    df = load_scores(json_filename)

    # 1. Select the scenario and the models
    if df['gamma'].nunique() > 1:
        df = df[np.isclose(df['gamma'], gamma)]
    if models is not None:
        df = df[df['model'].isin(models)]
    df = df.copy()
    df['Model'] = df['Model'].cat.remove_unused_categories()
    if df.empty:
        print("No rows left after filtering (check gamma / model names).")
        return

    n_reps = df['repetition'].nunique()
    print(f"Plotting {df['Model'].nunique()} models, gamma = {df['gamma'].iloc[0]}, "
          f"{n_reps} repetitions")
    print(df.groupby('Model', observed=True)[list(METRICS_CONFIG)].mean().round(4))

    metrics_to_plot = {k: v for k, v in METRICS_CONFIG.items() if k in df.columns}
    palette = dict(zip(df['Model'].cat.categories,
                       sns.color_palette('Set1', n_colors=len(df['Model'].cat.categories))))

    # 2. Plotting grid (3 rows x 2 columns; last slot holds the legend)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axes_flat = axes.flatten()

    # 3. One bar plot per metric
    for i, (metric, title) in enumerate(metrics_to_plot.items()):
        ax = axes_flat[i]
        # Drop excluded models for this panel only; their slot stays empty so
        # bars keep the same position and colour in every panel
        df_metric = df[~df['model'].isin(EXCLUDE_FROM_METRIC.get(metric, []))]
        sns.barplot(
            data=df_metric, x='Model', y=metric, hue='Model', palette=palette,
            legend=True, ax=ax, capsize=0.1, errorbar=('ci', 95),
            edgecolor='black', dodge=False,     # one full-width bar per model
        )
        ax.set_title('')
        ax.set_ylabel('')
        ax.set_xlabel(title, fontweight='normal', fontsize=12, labelpad=10)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        if metric == 'dist_preservation_score':
            ax.axhline(0, color='black', linewidth=0.8)     # 0 = same as mean imputation

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    # 4. Legend in the last slot
    legend_ax = axes_flat[5]
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center', title="Imputation Methods",
                     title_fontsize='12', fontsize='10', frameon=True,
                     shadow=True, facecolor='#f0f0f0')

    # 5. Remove unused slots (if fewer than 5 metrics)
    for j in range(len(metrics_to_plot), 5):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.show()


def plot_metrics_vs_gamma(json_filename, models=None, output='imputation_vs_gamma.png'):
    """
    For MNAR sensitivity results (several gamma values): each metric vs gamma,
    one line per model, with 95% CI bands. gamma = 0 is the MAR reference.
    """
    df = load_scores(json_filename)
    if models is not None:
        df = df[df['model'].isin(models)].copy()
        df['Model'] = df['Model'].cat.remove_unused_categories()
    if df['gamma'].nunique() < 2:
        print("Only one gamma in the file: nothing to plot against gamma.")
        return

    metrics_to_plot = {k: v for k, v in METRICS_CONFIG.items() if k in df.columns}
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 11))
    axes_flat = axes.flatten()
    for i, (metric, title) in enumerate(metrics_to_plot.items()):
        ax = axes_flat[i]
        sns.lineplot(data=df, x='gamma', y=metric, hue='Model', marker='o',
                     errorbar=('ci', 95), palette='Set1', ax=ax)
        ax.axvline(0, color='grey', linestyle=':', linewidth=1)   # MAR reference
        ax.set_xlabel(r'$\gamma$ (MNAR strength)')
        ax.set_ylabel('')
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

    axes_flat[5].axis('off')
    axes_flat[5].legend(handles, labels, loc='center', title="Imputation Methods",
                        fontsize='10', frameon=True, shadow=True, facecolor='#f0f0f0')
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_imputation_metrics('all_scores_MAR_Edist.json', gamma=0.0)
    # When the file contains several gamma values (MNAR sensitivity):
    # plot_metrics_vs_gamma('all_scores_MNAR.json')
