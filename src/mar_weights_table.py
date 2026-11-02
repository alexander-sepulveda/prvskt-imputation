import numpy as np


# Column layout of the table: (group name, [(LaTeX header, column index in the data), ...])
# The indices follow the order of `var_names` used to fit `mar_model`.
DEFAULT_GROUPS = [
    ("Material Representation", [("$L_1$", 0), ("$L_2$", 1), ("$L_3$", 2), ("$L_4$", 3)]),
    ("Solvent",                 [("Ratio", 4)]),
    ("Annealing",               [("Temp.", 5), ("Time", 6), ("Exp.", 7)]),
    ("Perovskite",              [("B. Gap", 8), ("Thick.", 9), ("Area", 10)]),
    ("Performance",             [("$V_{oc}$", 11), ("$J_{sc}$", 12), ("FF", 13), ("PCE", 14)]),
]


def _weights_as_array(weights, n_cols, col_names=None):
    """pyampute weights can be an array or a dict {index or name: weight}."""
    if isinstance(weights, dict):
        w = np.zeros(n_cols)
        for key, value in weights.items():
            j = col_names.index(key) if isinstance(key, str) else int(key)
            w[j] = value
        return w
    return np.asarray(weights, dtype=float)


def _fmt(value, decimals=2, zero_str="0."):
    """Two decimals by default; small non-zero values keep one significant digit."""
    if value == 0:
        return zero_str
    if abs(value) < 10 ** (-decimals):
        return f"{value:.{decimals + 1}f}"          # e.g. -0.006, 0.003
    return f"{value:.{decimals}f}"


def mar_weights_to_latex(mar_model, groups=DEFAULT_GROUPS, max_patterns=7,
                         col_names=None, decimals=2, zero_str="0.",
                         incomplete_str=None, show_freq=False, bold_max=False,
                         show_importance=True, weight_by_freq=False,
                         importance_label=None, importance_decimals=1,
                         caption=("Logistic Regression coefficients associated with the most "
                                  "frequent missingness patterns ($P_1$--$P_{%d}$)."),
                         label="tab:mar_weights", filename=None):
    """
    Build a LaTeX table (booktabs) with the weights of each pattern of a
    pyampute MultivariateAmputation object.

    Parameters
    ----------
    mar_model      : fitted pyampute MultivariateAmputation (uses mar_model.patterns).
    groups         : column groups and their (header, column index); see DEFAULT_GROUPS.
    max_patterns   : number of patterns to show (in the order of mar_model.patterns).
                     None = all patterns.
    col_names      : list of column names; only needed if weights or incomplete_vars
                     use names instead of indices.
    decimals       : number of decimals.
    zero_str       : how to print a weight equal to 0.
    incomplete_str : if given (e.g. "--"), the pattern's incomplete variables are
                     printed with this symbol instead of their (zero) weight, so they
                     can be told apart from predictors with a fitted weight of ~0.
    show_freq      : add a column with each pattern's relative frequency.
    bold_max       : bold the largest absolute weight of each row.
    show_importance: add a final row with, for each variable, the sum of the absolute
                     weights over the shown patterns, as a percentage of the total sum
                     of absolute weights (all variables, all shown patterns). The
                     percentages add up to 100.
    weight_by_freq : if True, each pattern's |w| is multiplied by its frequency f_k
                     before summing, so frequent patterns count more:
                     100 * sum_k f_k|w_kj| / sum_k sum_j f_k|w_kj|.
                     If False, every shown pattern counts equally.
    importance_label, importance_decimals : label and decimals of that row
                     (default label: "sum|w| (%)" or "sum f|w| (%)").
    caption, label : LaTeX caption and label ("%d" in caption -> number of patterns).
    filename       : if given, the table is also written to this .tex file.

    Returns
    -------
    str : LaTeX code. Requires \\usepackage{booktabs} in the preamble.
    """
    patterns = list(mar_model.patterns)
    if max_patterns is not None:
        patterns = patterns[:max_patterns]
    n_patterns = len(patterns)
    n_cols = max(j for _, cols in groups for _, j in cols) + 1

    all_cols = [(header, j) for _, cols in groups for header, j in cols]
    n_data_cols = len(all_cols)
    first_cols = 2 if show_freq else 1

    # ---------------- header ----------------
    col_spec = "l" + ("c" if show_freq else "")
    for _, cols in groups:
        col_spec += "r" * len(cols)

    group_line = ["", ""] if show_freq else [""]
    cmidrules = []
    start = first_cols + 1
    for name, cols in groups:
        end = start + len(cols) - 1
        group_line.append(rf"\multicolumn{{{len(cols)}}}{{c}}{{{name}}}")
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1

    header_line = ["Pattern"] + (["Freq."] if show_freq else []) + [h for h, _ in all_cols]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption % n_patterns if '%d' in caption else caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(group_line) + r" \\",
        " ".join(cmidrules),
        " & ".join(header_line) + r" \\",
        r"\midrule",
    ]

    # ---------------- body ----------------
    abs_sum = np.zeros(n_cols)                        # sum of |w| per variable
    for k, pattern in enumerate(patterns, start=1):
        w = _weights_as_array(pattern["weights"], n_cols, col_names)
        f_k = float(pattern.get("freq", np.nan)) if weight_by_freq else 1.0
        if np.isnan(f_k):
            raise ValueError(f"Pattern {k} has no 'freq'; cannot use weight_by_freq=True.")
        abs_sum += f_k * np.abs(w)
        incomplete = {col_names.index(v) if isinstance(v, str) else int(v)
                      for v in np.atleast_1d(pattern.get("incomplete_vars", []))}
        j_max = max((j for _, j in all_cols if j not in incomplete),
                    key=lambda j: abs(w[j]))

        cells = [str(k)]
        if show_freq:
            cells.append(f"{pattern.get('freq', np.nan):.3f}")
        for _, j in all_cols:
            if incomplete_str is not None and j in incomplete:
                cell = incomplete_str
            else:
                cell = _fmt(w[j], decimals, zero_str)
                if bold_max and j == j_max:
                    cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    # ---------------- relative importance row ----------------
    if show_importance:
        shown = [j for _, j in all_cols]
        total = abs_sum[shown].sum()
        if importance_label is None:
            importance_label = (r"$\sum_k f_k|w_k|$ (\%)" if weight_by_freq
                                else r"$\sum|w|$ (\%)")
        cells = [importance_label] + ([""] if show_freq else [])
        for j in shown:
            pct = 100 * abs_sum[j] / total if total > 0 else 0.0
            cells.append(f"{pct:.{importance_decimals}f}")
        lines += [r"\midrule", " & ".join(cells) + r" \\"]

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)

    if filename is not None:
        with open(filename, "w") as f:
            f.write(latex + "\n")
    return latex


# --------------------------- usage ---------------------------
# latex = mar_weights_to_latex(mar_model, max_patterns=7)
# print(latex)
#
# Variants:
# mar_weights_to_latex(mar_model, incomplete_str="--")          # mark missing vars of each pattern
# mar_weights_to_latex(mar_model, show_freq=True, bold_max=True) # add frequency, bold the main driver
# mar_weights_to_latex(mar_model, filename="mar_weights.tex")    # save to file, then \input{mar_weights}
# mar_weights_to_latex(mar_model, weight_by_freq=True)           # last row weighted by pattern frequency
