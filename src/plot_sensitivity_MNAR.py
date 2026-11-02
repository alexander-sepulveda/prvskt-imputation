# MNAR sensitivity analysis: performance metrics vs gamma (MNAR strength).
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from plot_results_MAR import load_scores, METRICS_CONFIG

# Fixed colour per model: seaborn Set1, same as imputation_results.png. Marker and
# line style give a second, colour-independent cue for print / colour-blind readers.
STYLE = {
    'Mean_Baseline':         {'color': '#e41a1c', 'marker': 'o', 'linestyle': '-'},
    'MissForest_Classic_50': {'color': '#4daf4a', 'marker': 's', 'linestyle': '--'},
    'MissForest_Extra_50':   {'color': '#984ea3', 'marker': '^', 'linestyle': '-.'},
    'MICE_KNN_5':            {'color': '#ff7f00', 'marker': 'D', 'linestyle': ':'},
}
FALLBACK_STYLE = {'color': '#6b6b6b', 'marker': 'v', 'linestyle': '-'}


def keep_complete_repetitions(df):
    """
    Keep only repetitions that contain every (model, gamma) combination, so all
    gamma values are compared on the same paired set of repetitions.
    """
    n_cells = df['model'].nunique() * df['gamma'].nunique()
    counts = df.groupby('repetition').size()
    complete = counts[counts == n_cells].index
    dropped = sorted(set(counts.index) - set(complete))
    if dropped:
        print(f"  >> Dropping incomplete repetitions {dropped} "
              f"(missing some gamma/model results).")
    return df[df['repetition'].isin(complete)].copy()


def mean_ci(values, level=0.95):
    """Mean and t-based confidence half-width over repetitions."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    m = values.mean()
    if n < 2:
        return m, 0.0
    half = stats.t.ppf(0.5 + level / 2, n - 1) * values.std(ddof=1) / np.sqrt(n)
    return m, half


def summarize(df, metrics):
    """Long table: model, gamma, metric, mean, ci_low, ci_high, n."""
    rows = []
    for (model, gamma), g in df.groupby(['model', 'gamma']):
        for metric in metrics:
            m, h = mean_ci(g[metric])
            rows.append({'model': model, 'gamma': gamma, 'metric': metric,
                         'mean': m, 'ci_low': m - h, 'ci_high': m + h, 'n': len(g)})
    return pd.DataFrame(rows)


def paired_difference(df, metrics, reference_gamma=0.0):
    """Per repetition: metric(gamma) - metric(reference_gamma), same model."""
    ref = df[np.isclose(df['gamma'], reference_gamma)]
    ref = ref.set_index(['repetition', 'model'])[metrics]
    out = df.set_index(['repetition', 'model']).copy()
    out[metrics] = out[metrics] - ref.reindex(out.index).values
    return out.reset_index()


def plot_vs_gamma(summary, df, metrics, output, ylabel_suffix='', reference_line=None):
    """Grid of panels (one per metric): mean ± 95% CI vs gamma, one line per model."""
    models = [m for m in STYLE if m in df['model'].unique()]
    models += [m for m in df['model'].unique() if m not in models]
    labels = df.drop_duplicates('model').set_index('model')['Model'].astype(str)
    gammas = sorted(df['gamma'].unique())

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(11, 11))
    axes_flat = axes.flatten()
    for i, metric in enumerate(metrics):
        ax = axes_flat[i]
        for model in models:
            s = summary[(summary['model'] == model) & (summary['metric'] == metric)]
            s = s.sort_values('gamma')
            st = STYLE.get(model, FALLBACK_STYLE)
            ax.fill_between(s['gamma'], s['ci_low'], s['ci_high'],
                            color=st['color'], alpha=0.15, linewidth=0)
            ax.plot(s['gamma'], s['mean'], color=st['color'], marker=st['marker'],
                    linestyle=st['linestyle'], linewidth=2, markersize=6,
                    markeredgecolor='white', markeredgewidth=1, label=labels[model])
        ax.axvline(0, color='grey', linestyle=':', linewidth=1)          # MAR reference
        if reference_line is not None or metric == 'dist_preservation_score':
            ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(gammas)
        ax.set_xlabel(r'$\gamma$ (MNAR strength)')
        ax.set_ylabel('')
        # Caption below the panel (under the gamma axis label), as in imputation_results.png
        ax.set_title(METRICS_CONFIG[metric] + ylabel_suffix, fontsize=12, y=0, pad=-50,
                     verticalalignment='top')
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', alpha=0.4)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        if i == 0:
            handles, leg_labels = ax.get_legend_handles_labels()

    legend_ax = axes_flat[5]
    legend_ax.axis('off')
    n_reps = df['repetition'].nunique()
    legend_ax.legend(handles, leg_labels, loc='center',
                     title=f"Imputation Methods\n(mean ± 95% CI, {n_reps} repetitions)",
                     title_fontsize='11', fontsize='10', frameon=True, facecolor='#f5f5f5')
    for j in range(len(metrics), 5):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches='tight')
    print(f"  >> Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_file', nargs='?', default='scores_MNAR.json')
    parser.add_argument('--models', nargs='+', default=None,
                        help="raw model names to include, e.g. Mean_Baseline MICE_KNN_5")
    parser.add_argument('--prefix', default='mnar_sensitivity',
                        help="prefix for the output files")
    parser.add_argument('--show', action='store_true', help="also open the figures")
    args = parser.parse_args()

    df = load_scores(args.json_file)
    if args.models is not None:
        df = df[df['model'].isin(args.models)].copy()
    if df['gamma'].nunique() < 2:
        raise SystemExit("Only one gamma in the file: nothing to plot against gamma.")
    df = keep_complete_repetitions(df)
    metrics = [m for m in METRICS_CONFIG if m in df.columns]

    # 1. Absolute metrics vs gamma
    summary = summarize(df, metrics)
    summary.to_csv(f'{args.prefix}_summary.csv', index=False)
    print(summary.pivot_table(index=['metric', 'model'], columns='gamma',
                              values='mean').round(4))
    plot_vs_gamma(summary, df, metrics, f'{args.prefix}.png')

    # 2. Paired change with respect to MAR (gamma = 0), per repetition
    if np.isclose(df['gamma'], 0).any():
        diff = paired_difference(df, metrics, reference_gamma=0.0)
        summary_diff = summarize(diff, metrics)
        summary_diff.to_csv(f'{args.prefix}_delta_vs_MAR.csv', index=False)
        plot_vs_gamma(summary_diff, diff, metrics, f'{args.prefix}_delta_vs_MAR.png',
                      ylabel_suffix=r'  [$\Delta$ vs $\gamma=0$]', reference_line=0)

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
