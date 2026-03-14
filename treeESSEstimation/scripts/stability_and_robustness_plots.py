"""
Stability and Robustness Plots and Summary Tables
==================================================

Produces plots and LaTeX summary tables for two experiments:

**Robustness** — how consistently an estimator scores when the MCMC chain
is split into *k* equal fragments. Data is loaded from a single flat CSV
covering all datasets and runs.

**Stability** — how stable ESS estimates are when the chain is thinned
(subsampled at different rates). Data is loaded from a single flat CSV.

Both experiments share the same ``ESTIMATORS`` list.  Comment out any
estimators that should not appear.

Comment in or out the calls in ``main()`` to select what you want.

File formats
------------
Robustness CSV (tab-separated)::

    ds  run  estimator  k  sum  <fragment columns...>

Stability CSV (tab-separated)::

    ds  run  estimator  <thinning-level columns...>

Expected file locations (configurable via the path constants below)::

    {base}/robustness/robustness{id}k-{method}.csv
    {base}/stability.csv

Display names and colours
-------------------------
Estimator names are used as labels directly.  For consistent display names
and fixed colours across multiple figures, define a mapping dict and pass
it to the plotting/table functions::

    ESTIMATOR_DISPLAY = {
        "CLADE_INDICATOR": {"name": "Clade Indicator", "color": "#1f77b4"},
        ...
    }
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


# ==========================================================
# VISUAL STYLE
# ==========================================================

FONT_SIZE: int = 12

mpl.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE,
    "ytick.labelsize":  FONT_SIZE,
    "legend.fontsize":  FONT_SIZE,
})


# ==========================================================
# FILE CONFIGURATION  (edit here)
# ==========================================================

# Root directory containing data subdirectories.
DEFAULT_BASE_PATH: Path = Path(__file__).parent.parent / "example_data"

# Method / dataset-type token embedded in the robustness file name.
DEFAULT_METHOD: str = "DS"

# Robustness file-name pattern.  Placeholders: {base}, {id}, {method}.
ROBUSTNESS_FILE_PATTERN: str = "{base}/robustness/robustness-{id}k-{method}.csv"

# Stability data file.  Placeholder: {base}.
STABILITY_FILE_PATTERN: str = "{base}/stability/stability.csv"


# ==========================================================
# DATASET RANGE
# ==========================================================

# Dataset index range (inclusive).
DEFAULT_DS_START: int = 1
DEFAULT_DS_END: int = 11


# ==========================================================
# ROBUSTNESS DEFAULTS
# ==========================================================

# Chain length in thousands of trees used in the robustness experiment.
DEFAULT_ROBUSTNESS_ID: int = 10

# Which run to show in the bar plot (summary table uses an explicit argument).
DEFAULT_ROBUSTNESS_RUN: int = 1


# ==========================================================
# STABILITY DEFAULTS
# ==========================================================

# Labels for the thinning levels as they appear in the stability bar plot.
# Must match the column order in the stability CSV.
STABILITY_THINNING_LABELS: list[str] = [
    "All trees",
    "Every 2nd",
    "Every 4th",
    "Every 8th",
    "Every 16th",
]


# ==========================================================
# ESTIMATORS
# ==========================================================

# List the estimators to include in plots and tables.
# Comment out any that should not be displayed.
ESTIMATORS: list[str] = [
    'CLADE_INDICATOR',
    ## 'CLADE_INDICATOR_AVG',
    ## 'CLADE_INDICATOR_MEDIAN',
    'FrechetCorrelationESS',
    'LogP1',
    'LogP0',
    'AvgRF1',
    'AvgRF0',
    'MAP_RF1',
    'MAP_RF0',
    ## 'MAP_RNNI1',
    ## 'MAP_RNNI0',
    'RAND_RF_MEDIAN',
    'RAND_RF_MIN',
    ## 'RAND_RNNI_MEDIAN',
    ## 'RAND_RNNI_MIN',
    'BinomESS_Bayesian',
    'BinomESS_ML',
]


# ==========================================================
# OUTPUT CONFIGURATION
# ==========================================================

DEFAULT_OUTPUT_DIR: Path = Path(__file__).parent


# ==========================================================
# LATEX TABLE HELPER
# ==========================================================

def _dict_to_latex_table(
        results: dict,
        caption: str = "",
) -> str:
    """
    Render a nested results dict as a LaTeX figure containing a coloured table.

    Dict structure::

        { dataset_key: { estimator_name: value } }

    *value* may be a plain float (stability: normalised std) or a
    ``(mean, std)`` tuple (robustness).  The special dataset keys ``"avg"``
    and ``"rank"`` are rendered without colour boxes.

    Colour coding (green / orange / red) reflects deviation from zero using
    fixed thresholds: ≤ 0.05, 0.05–0.1, > 0.1.

    Args:
        results: Nested dict of results as described above.
        caption: LaTeX caption string for the figure environment.

    Returns:
        LaTeX string containing a ``figure`` environment with the table.
    """
    fmt = "{:.2f}"
    datasets   = sorted(results.keys(), key=lambda x: (isinstance(x, str), x))
    estimators = list({est for d in results.values() for est in d.keys()})
    # Preserve ESTIMATORS order for rows that appear in the global list.
    estimators = sorted(estimators,
                        key=lambda e: ESTIMATORS.index(e) if e in ESTIMATORS else len(ESTIMATORS))

    table = pd.DataFrame(index=estimators, columns=datasets)

    def _colorbox(val: float, bg: str) -> str:
        return f"\\colorbox{{{bg}!25}}{{{fmt.format(val)}}}"

    def _cellcolor(val: float, bg: str) -> str:
        return f"\\cellcolor{{{bg}!25}}"

    def _color_class(val: float) -> str:
        a = abs(val)
        if a <= 0.05:  return "green"
        if a <= 0.10:  return "orange"
        return "red"

    for ds, est_dict in results.items():
        for esti, entry in est_dict.items():
            if isinstance(entry, tuple):
                mean, std = entry
                mean_str = _colorbox(mean, _color_class(mean))
                std_str  = _colorbox(std,  _color_class(std))
                table.loc[esti, ds] = f"${mean_str} \\pm {std_str}$"
            else:
                val = entry
                if ds == "avg":
                    table.loc[esti, ds] = fmt.format(val)
                elif ds == "rank":
                    table.loc[esti, ds] = str(int(val))
                else:
                    cc = _cellcolor(val, _color_class(val))
                    table.loc[esti, ds] = f"${cc} \\pm {fmt.format(val)}$"

    table.dropna(inplace=True)
    latex_str = table.to_latex(
        escape=False,
        column_format="l" + "c" * len(datasets),
    )
    indented = "\n".join("        " + line for line in latex_str.splitlines())
    return (
        "\\begin{figure}[h]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{fig:placeholder}\n"
        "\\resizebox{1.3\\textwidth}{!}{\n"
        f"{indented}\n"
        "}\n"
        "\\end{figure}"
    )


# ==========================================================
# ROBUSTNESS
# ==========================================================

def plot_robustness(
        id: int = DEFAULT_ROBUSTNESS_ID,
        run: int = DEFAULT_ROBUSTNESS_RUN,
        estimators: list[str] = ESTIMATORS,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        method: str = DEFAULT_METHOD,
        base_path: Path = DEFAULT_BASE_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Stacked bar chart of fragment ESS contributions per estimator per dataset.

    One page per dataset is saved in a multi-page PDF.  Only *run* is shown.

    Args:
        id:         Chain length in thousands of trees.
        run:        Run index to display.
        estimators: Ordered list of estimator names to include.
        ds_start:   First dataset index (inclusive).
        ds_end:     Last dataset index (inclusive).
        method:     Method token used in the file name.
        base_path:  Root data directory.
        output_dir: Directory where the PDF is written.
    """
    path = ROBUSTNESS_FILE_PATTERN.format(base=base_path, id=id, method=method)
    df   = pd.read_csv(path, sep="\t")

    pdf_path = output_dir / f"robustness-{id}k-run{run}-{method}.pdf"
    with PdfPages(pdf_path) as pdf:
        for ds in range(ds_start, ds_end + 1):
            print(f"  Plotting {method}{ds}")
            cur = (
                df[(df["ds"] == ds) & (df["run"] == run)]
                .drop(columns=["ds", "sum", "run"])
                .groupby(["estimator", "k"])
                .mean()
            )
            cur = cur[cur.index.get_level_values("estimator").isin(estimators)]
            cur = cur.reindex(estimators, level="estimator")

            ax = cur.plot.bar(stacked=True, figsize=(10, 5))
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Fragments")

            labels    = [t.get_text() for t in ax.get_xticklabels()]
            new_ticks = np.arange(5, len(labels), 10)
            ax.set_xticks(new_ticks,
                          labels=[labels[i].split(",")[0][1:] for i in new_ticks])
            ax.set_axisbelow(True)
            ax.grid(axis="y")

            plt.xlabel("")
            plt.ylabel("Estimated Tree ESS")
            plt.suptitle(f"Robustness Experiment with {id}k Trees — {method}{ds}")
            plt.tight_layout()

            pdf.savefig()
            plt.cla()
            plt.close()

    print(f"Saved: {pdf_path}")


def print_latex_robustness_table(
        id: int = DEFAULT_ROBUSTNESS_ID,
        run: int = DEFAULT_ROBUSTNESS_RUN,
        estimators: list[str] = ESTIMATORS,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        method: str = DEFAULT_METHOD,
        base_path: Path = DEFAULT_BASE_PATH,
) -> None:
    """
    Print a LaTeX robustness summary table for one chain length and run.

    Each cell shows ``mean ± std`` of normalised pairwise fragment
    differences, colour-coded by deviation magnitude.

    Args:
        id:         Chain length in thousands of trees.
        run:        Run index to use.
        estimators: Ordered list of estimator names to include.
        ds_start:   First dataset index (inclusive).
        ds_end:     Last dataset index (inclusive).
        method:     Method token used in the file name.
        base_path:  Root data directory.
    """
    path = ROBUSTNESS_FILE_PATTERN.format(base=base_path, id=id, method=method)
    df   = pd.read_csv(path, sep="\t")

    results: dict = {}
    for ds in range(ds_start, ds_end + 1):
        cur = (
            df[(df["ds"] == ds) & (df["run"] == run)]
            .drop(columns=["ds", "sum", "run"])
        )
        results[ds] = {}
        for est in estimators:
            tmp = cur[cur["estimator"] == est]
            if tmp.empty:
                print(f"  No data for estimator {est} in {method}{ds}")
                continue
            sums = tmp.drop(columns=["estimator", "k"]).sum(axis=1, skipna=True)
            diffs = sums.values[np.newaxis:] - sums.values[:, np.newaxis]
            relevant = np.abs(diffs[np.triu_indices(len(sums), k=1)])
            scale = sums.max()
            results[ds][est] = (relevant.mean() / scale, relevant.std() / scale)

    print(_dict_to_latex_table(
        results,
        caption=f"Robustness summary — {id}k trees, run {run}, {method}",
    ))


# ==========================================================
# STABILITY
# ==========================================================

def plot_stability(
        estimators: list[str] = ESTIMATORS,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        thinning_labels: list[str] = STABILITY_THINNING_LABELS,
        base_path: Path = DEFAULT_BASE_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Grouped bar chart of estimated ESS at different thinning levels per dataset.

    One page per dataset is saved in a multi-page PDF.

    Args:
        estimators:      Ordered list of estimator names to include.
        ds_start:        First dataset index (inclusive).
        ds_end:          Last dataset index (inclusive).
        thinning_labels: Legend labels for each thinning column, in column order.
        base_path:       Root data directory.
        output_dir:      Directory where the PDF is written.
    """
    path = STABILITY_FILE_PATTERN.format(base=base_path)
    df   = pd.read_csv(path, sep="\t")

    pdf_path = output_dir / "stability.pdf"
    with PdfPages(pdf_path) as pdf:
        for ds in range(ds_start, ds_end + 1):
            cur = (
                df[df["ds"] == ds]
                .drop(columns=["ds", "run"])
                .groupby("estimator")
                .mean()
            )
            cur = cur[cur.index.isin(estimators)]
            cur = cur.reindex(estimators)

            fig, ax = plt.subplots(figsize=(10, 5))
            cur.plot.bar(stacked=False, ax=ax)
            ax.legend(thinning_labels, title="Thinning",
                      loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set_axisbelow(True)
            ax.grid(axis="y")

            plt.xlabel("")
            plt.ylabel("Estimated Tree ESS")
            plt.suptitle(f"Subsampling Experiment — DS {ds}")
            plt.tight_layout()

            pdf.savefig()
            plt.cla()
            plt.close()

    print(f"Saved: {pdf_path}")


def print_latex_stability_table(
        estimators: list[str] = ESTIMATORS,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        base_path: Path = DEFAULT_BASE_PATH,
) -> None:
    """
    Print a LaTeX stability summary table.

    Each cell shows the normalised std of pairwise thinning-level differences,
    colour-coded by magnitude.  An ``avg`` column and a ``rank`` column
    (by avg score) are appended.

    Args:
        estimators: Ordered list of estimator names to include.
        ds_start:   First dataset index (inclusive).
        ds_end:     Last dataset index (inclusive).
        base_path:  Root data directory.
    """
    path = STABILITY_FILE_PATTERN.format(base=base_path)
    df   = pd.read_csv(path, sep="\t")

    results: dict = {}
    for ds in range(ds_start, ds_end + 1):
        cur = df[df["ds"] == ds].drop(columns=["ds", "run"])
        results[ds] = {}
        for est in estimators:
            tmp = cur[cur["estimator"] == est]
            if tmp.empty:
                print(f"  No data for estimator {est} in DS{ds}")
                continue
            max_ess    = tmp.drop(columns="estimator").values.max()
            avg_values = tmp.groupby("estimator").mean().values[0]
            diffs      = avg_values[:, np.newaxis] - avg_values[np.newaxis, :]
            relevant   = diffs[np.triu_indices(len(avg_values), k=1)]
            results[ds][est] = relevant.std() / max_ess

    # Average column across all datasets.
    ds_count = ds_end - ds_start + 1
    results["avg"] = {}
    for est in estimators:
        vals = [results[ds].get(est) for ds in range(ds_start, ds_end + 1)
                if results[ds].get(est) is not None]
        if vals:
            results["avg"][est] = sum(vals) / ds_count

    # Rank column by average score (lowest = best).
    avgs = results["avg"]
    results["rank"] = {
        est: rank
        for rank, est in enumerate(sorted(avgs, key=avgs.__getitem__), start=1)
    }

    print(_dict_to_latex_table(
        results,
        caption="Stability summary — normalised pairwise thinning differences",
    ))


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == '__main__':
    # Comment in/out the outputs you want to produce.

    # Robustness bar plots (one page per dataset, saved to PDF):
    plot_robustness()

    # Stability bar plots (one page per dataset, saved to PDF):
    plot_stability()

    # LaTeX robustness summary table:
    # print_latex_robustness_table()

    # LaTeX stability summary table:
    # print_latex_stability_table()
    pass
