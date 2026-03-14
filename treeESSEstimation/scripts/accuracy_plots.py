"""
ESS Estimator Accuracy Plots and Error Tables
===============================================

Reads per-chain accuracy data (estimated vs. true Tree ESS across sample
sizes) and produces three types of output:

* **Line plots** — estimated ESS against true ESS, absolute or relative
  (one page per dataset, collected in a multi-page PDF).
* **Box plots** — relative mean error across ACT values and datasets.
* **LaTeX tables** — ME / MSE per dataset, or normalised rank sums across
  ACT values.

Comment in or out the calls in ``main()`` to select what you want.

File format
-----------
Tab-separated CSV with columns::

    chain   estimator   10   20   30   ...   1000

``chain`` identifies the MCMC run; ``estimator`` names the ESS method;
the remaining columns give estimated ESS at each sample size (ACT value).

Expected file-name pattern (configurable via ``FILE_PATTERN``)::

    {base}/{method}/ACT{act}/{prefix}{nbr}-{method}-ACT{act}.csv

Example::

    ../data/accuracy/Noisy/ACT10/DS7-Noisy-ACT10.csv

Display names and colours
-------------------------
Without a mapping, estimator names are used as labels and matplotlib
assigns colours automatically.  For consistent names and fixed colours
across multiple figures, define a dict and pass it to the plotting
functions::

    ESTIMATOR_DISPLAY = {
        "CLADE_INDICATOR": {"name": "Clade Indicator", "color": "#1f77b4"},
        "LogP1":           {"name": "log P (bounded)", "color": "#ff7f0e"},
        ...
    }
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
from pathlib import Path


# ==========================================================
# VISUAL STYLE
# ==========================================================

FONT_SIZE: int = 14

mpl.rcParams.update({
    "font.size":        FONT_SIZE,
    "axes.titlesize":   FONT_SIZE,
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE,
    "ytick.labelsize":  FONT_SIZE,
    "legend.fontsize":  FONT_SIZE,
})


# ==========================================================
# FILE CONFIGURATION  (edit here or pass kwargs to functions)
# ==========================================================

# Base directory containing accuracy data.
DEFAULT_BASE_PATH: Path = Path("..") / "example_data" / "accuracy"

# Simulation method embedded in file names (e.g. "Noisy", "RNNI").
DEFAULT_SIMULATION_METHOD: str = "RNNI"

# Prefix for dataset file names (e.g. "DS" → "DS7-...", "YuleRep" → "YuleRep7-...").
DEFAULT_DS_PREFIX: str = "DS"

# Dataset index range (inclusive).  Set start and end to select a subset,
# e.g. DEFAULT_DS_START = 3, DEFAULT_DS_END = 4 to process only DS3 and DS4.
DEFAULT_DS_START: int = 1
DEFAULT_DS_END: int = 11

# File-name pattern.  Placeholders: {base}, {method}, {act}, {prefix}, {nbr}.
FILE_PATTERN: str = "{base}/{method}/ACT{act}/{prefix}{nbr}-{method}-ACT{act}.csv"


# ==========================================================
# ACT VALUES
# ==========================================================

# ACT values used in multi-ACT tables and box plots.
# Adjust to match the values present in your data.
DEFAULT_ACT_LIST: list[int] = [2, 5, 10, 25, 50, 75, 100]


# ==========================================================
# ESTIMATORS
# ==========================================================

# List the estimators to include in plots and tables.
# Comment out any that should not be displayed.
ESTIMATORS: list[str] = [
    'CLADE_INDICATOR',
    'FrechetCorrelationESS',
    'LogP1',
    'LogP0',
    'AvgRF1',
    'AvgRF0',
    'MAP_RF1',
    'MAP_RF0',
    'RAND_RF_MEDIAN',
    'RAND_RF_MIN',
    # 'BinomESS_Bayesian',
]


# ==========================================================
# OUTPUT CONFIGURATION
# ==========================================================

DEFAULT_OUTPUT_DIR: Path = Path(__file__).parent


# ==========================================================
# DATA LOADING AND PROCESSING
# ==========================================================

def load_dataset(
        act: int,
        nbr: int,
        method: str = DEFAULT_SIMULATION_METHOD,
        prefix: str = DEFAULT_DS_PREFIX,
        base_path: Path = DEFAULT_BASE_PATH,
        file_pattern: str = FILE_PATTERN,
) -> pd.DataFrame:
    """
    Load one accuracy dataset from disk.

    Args:
        act:          ACT value embedded in the file name.
        nbr:          Dataset / repetition index.
        method:       Simulation method string used in the file name.
        prefix:       Dataset file-name prefix (e.g. ``"DS"`` or ``"YuleRep"``).
        base_path:    Root directory for accuracy data.
        file_pattern: File-name format string with placeholders
                      ``{base}``, ``{method}``, ``{act}``, ``{prefix}``,
                      ``{nbr}``.

    Returns:
        DataFrame with columns ``chain``, ``estimator``, and one column
        per sample size.
    """
    path = file_pattern.format(
        base=base_path, method=method, act=act, prefix=prefix, nbr=nbr,
    )
    return pd.read_csv(path, delimiter='\t')


def compute_mean_by_estimator(data: pd.DataFrame) -> pd.DataFrame:
    """
    Average estimated ESS values across chains, grouped by estimator.

    Args:
        data: DataFrame as returned by [load_dataset].

    Returns:
        DataFrame indexed by estimator with integer sample-size columns.
    """
    mean_df = (
        data
        .drop(columns=["chain"])
        .groupby("estimator", as_index=True)
        .mean()
    )
    return mean_df.rename(
        columns={col: int(col) for col in mean_df.columns if col != "estimator"}
    )


def compute_errors(mean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append mean error, mean squared error, and relative mean error columns.

    Errors are computed over all sample-size columns (10 to 1000) by
    comparing estimated ESS against the true sample size (the column index).

    Args:
        mean_df: DataFrame as returned by [compute_mean_by_estimator].

    Returns:
        The same DataFrame with ``ME``, ``MSE``, and ``rel_ME`` columns added.
    """
    error_cols = {"estimator", "ME", "MSE", "rel_ME"}

    def _errors(row):
        x_vals = np.array(
            [c for c in mean_df.columns if c not in error_cols],
            dtype=int,
        )
        y_vals = row.values.astype(float)
        error = y_vals - x_vals
        return pd.Series({
            "ME":     np.mean(error),
            "MSE":    np.mean(error ** 2),
            "rel_ME": np.mean(error / x_vals),
        })

    mean_df[["ME", "MSE", "rel_ME"]] = mean_df.apply(_errors, axis=1)
    return mean_df


def compute_all_errors(
        act: int,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        method: str = DEFAULT_SIMULATION_METHOD,
        prefix: str = DEFAULT_DS_PREFIX,
        base_path: Path = DEFAULT_BASE_PATH,
        file_pattern: str = FILE_PATTERN,
) -> pd.DataFrame:
    """
    Load and compute errors across a range of datasets for one ACT value.

    Args:
        act:          ACT value to process.
        ds_start:     First dataset index (inclusive).
        ds_end:       Last dataset index (inclusive).
        method:       Simulation method string.
        prefix:       Dataset file-name prefix.
        base_path:    Root directory for accuracy data.
        file_pattern: File-name format string.

    Returns:
        DataFrame with columns ``estimator``, ``dataset``, ``ME``, ``MSE``,
        and ``rel_ME``, with one block of rows per dataset.
    """
    all_errors = []
    for nbr in range(ds_start, ds_end + 1):
        data = load_dataset(act, nbr, method, prefix, base_path, file_pattern)
        mean_df = compute_mean_by_estimator(data)
        mean_df = compute_errors(mean_df)

        cur = mean_df[["ME", "MSE", "rel_ME"]].copy()
        cur["estimator"] = mean_df.index
        cur["dataset"] = nbr
        all_errors.append(cur)

    return pd.concat(all_errors, ignore_index=True).dropna()


# ==========================================================
# LATEX TABLE HELPERS
# ==========================================================

def format_cells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format ME and MSE values as LaTeX strings, bolding the best per dataset.

    Each cell combines both metrics as ``"ME / MSE"``.  The estimator with
    the smallest |ME| and the smallest MSE within each dataset are bolded
    independently.

    Args:
        df: DataFrame as returned by [compute_all_errors], containing
            ``dataset``, ``ME``, and ``MSE`` columns.

    Returns:
        The same DataFrame with a ``cell`` column appended.
    """
    cells = []
    for _, row in df.iterrows():
        group = df[df["dataset"] == row["dataset"]]
        min_me  = group["ME"].abs().min()
        min_mse = group["MSE"].min()
        me_str  = (f"\\textbf{{{row['ME']:.2f}}}"  if abs(row["ME"])  == min_me  else f"{row['ME']:.2f}")
        mse_str = (f"\\textbf{{{row['MSE']:.2f}}}" if row["MSE"] == min_mse else f"{row['MSE']:.2f}")
        cells.append(f"{me_str} / {mse_str}")
    df["cell"] = cells
    return df


def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-dataset MSE and absolute-ME rank columns (rank 1 = best).

    Args:
        df: DataFrame with ``dataset``, ``ME``, and ``MSE`` columns.

    Returns:
        The same DataFrame with ``ME-rank`` and ``MSE-rank`` columns added.
    """
    df["MSE-rank"] = df.groupby("dataset")["MSE"].rank(method="min", ascending=True)
    df["abs_ME"]   = df["ME"].abs()
    df["ME-rank"]  = df.groupby("dataset")["abs_ME"].rank(method="min", ascending=True)
    df.drop(columns="abs_ME", inplace=True)
    return df


def summarize_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate rank sums across datasets and normalise to [0, 1].

    The best estimator receives a normalised score of 0; the worst receives 1.
    Best values are wrapped in ``\\textbf{}`` for LaTeX output.

    Args:
        df: DataFrame with ``estimator``, ``ME-rank``, and ``MSE-rank``
            columns, as returned by [compute_ranks].

    Returns:
        DataFrame with one row per estimator and columns for raw rank sums,
        normalised scores, and LaTeX-formatted strings.
    """
    rank_sums = (
        df.groupby("estimator")[["ME-rank", "MSE-rank"]]
        .sum()
        .reset_index()
    )

    for col in ["ME-rank", "MSE-rank"]:
        lo, hi = rank_sums[col].min(), rank_sums[col].max()
        rank_sums[col + "-norm"] = round((rank_sums[col] - lo) / (hi - lo), 2)

    for col in ["ME-rank-norm", "MSE-rank-norm"]:
        lo = rank_sums[col].min()
        rank_sums[col + "-latex"] = rank_sums[col].apply(
            lambda x: f"\\textbf{{{x:.2f}}}" if x == lo else f"{x:.2f}"
        )

    return rank_sums


def build_latex_error_table(df: pd.DataFrame, rank_sums: pd.DataFrame) -> str:
    """
    Build a LaTeX table string of ME / MSE cells with rank summary columns.

    Args:
        df:        DataFrame with ``estimator``, ``dataset``, and ``cell``
                   columns, as returned by [format_cells].
        rank_sums: DataFrame as returned by [summarize_ranks].

    Returns:
        LaTeX table string suitable for inclusion in a document.
    """
    pivot = df.pivot(index="estimator", columns="dataset", values="cell")
    pivot = pivot.merge(
        rank_sums[["estimator", "ME-rank-norm-latex", "MSE-rank-norm-latex"]],
        on="estimator",
        how="left",
    )
    return pivot.to_latex(
        escape=False,
        column_format="l" + "c" * (len(pivot.columns) - 1),
        index=False,
    )


# ==========================================================
# PLOTS
# ==========================================================

def plot_estimated_ess(
        act: int,
        relative: bool = False,
        estimators: list[str] = ESTIMATORS,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        method: str = DEFAULT_SIMULATION_METHOD,
        prefix: str = DEFAULT_DS_PREFIX,
        base_path: Path = DEFAULT_BASE_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Plot estimated Tree ESS against true ESS for each dataset.

    Each dataset is saved as one page in a multi-page PDF.  The reference
    line represents a perfect estimator: the diagonal in absolute mode,
    zero in relative mode.

    Args:
        act:        ACT value, used for data loading and file naming.
        relative:   If ``True``, plot ``(estimated − true) / true`` instead
                    of raw estimated values.
        estimators: Ordered list of estimator names to include.
        ds_start:   First dataset index (inclusive).
        ds_end:     Last dataset index (inclusive).
        method:     Simulation method string.
        prefix:     Dataset file-name prefix.
        base_path:  Root directory for accuracy data.
        output_dir: Directory where the PDF is written.
    """
    suffix   = "-rel" if relative else ""
    pdf_path = output_dir / f"{method}-ACT{act}{suffix}.pdf"
    x_vals   = np.arange(10, 1001, 10)

    with PdfPages(pdf_path) as pdf:
        for nbr in range(ds_start, ds_end + 1):
            data    = load_dataset(act, nbr, method, prefix, base_path)
            mean_df = compute_mean_by_estimator(data)

            plt.figure(figsize=(11, 6))
            lines, labels = [], []

            ref, = plt.plot(
                [0, 1000], [0, 0] if relative else [0, 1000],
                linestyle="dotted", color="black",
            )
            lines.append(ref)
            labels.append("# i.i.d. sampled trees")

            for estimator in estimators:
                y_raw  = mean_df.loc[estimator].values
                y_vals = (y_raw - x_vals) / x_vals if relative else y_raw
                line, = plt.plot(x_vals, y_vals, label=estimator)
                lines.append(line)
                labels.append(estimator)

            if relative:
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
                plt.ylabel("Relative Estimated Tree ESS")
            else:
                plt.ylabel("Estimated Tree ESS")
                plt.ylim(0, 1100)

            plt.xlabel("#iid Sampled Trees (≈ Chain length / ACT)")
            plt.title(f"Estimated Tree ESS ({method}, ACT={act}, {prefix}{nbr})")
            plt.xlim(0, 1000)
            plt.legend(lines, labels, loc="center left",
                       bbox_to_anchor=(1.02, 0.5), title="Estimator")
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"Saved: {pdf_path}")


def plot_relative_error_boxplot(
        act_list: list[int] = DEFAULT_ACT_LIST,
        estimators: list[str] = ESTIMATORS,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        method: str = DEFAULT_SIMULATION_METHOD,
        prefix: str = DEFAULT_DS_PREFIX,
        base_path: Path = DEFAULT_BASE_PATH,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """
    Box plot of relative mean error grouped by ACT value.

    The y-axis uses a symmetric log scale so the region near zero is legible
    while large outliers remain visible.

    Args:
        act_list:   ACT values to include as x-axis groups.
        estimators: Ordered list of estimator names to include.
        ds_start:   First dataset index (inclusive).
        ds_end:     Last dataset index (inclusive).
        method:     Simulation method string.
        prefix:     Dataset file-name prefix.
        base_path:  Root directory for accuracy data.
        output_dir: Directory where the PDF is written.
    """
    all_rows = []
    for act in act_list:
        df = compute_all_errors(act, ds_start, ds_end, method, prefix, base_path)
        df = df[df["estimator"].isin(estimators)].copy()
        df["estimator"] = pd.Categorical(df["estimator"], categories=estimators, ordered=True)
        df["ACT"] = act
        all_rows.append(df[["rel_ME", "estimator", "ACT"]])

    combined = pd.concat(all_rows, ignore_index=True)

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(
        data=combined, x="ACT", y="rel_ME", hue="estimator",
        linewidth=0.4, fliersize=0.5,
    )
    ax.axhline(0, color=".3", dashes=(2, 2), zorder=0)
    plt.yscale("symlog", linthresh=0.1)
    plt.ylabel("Mean Relative Error")
    plt.legend(title="Estimator", bbox_to_anchor=(1.05, 1),
               loc="upper left", borderaxespad=0.)
    plt.tight_layout()

    pdf_path = output_dir / f"summary_relME_{method}.pdf"
    plt.savefig(pdf_path, dpi=300, format="pdf")
    print(f"Saved: {pdf_path}")


# ==========================================================
# LATEX TABLES
# ==========================================================

def print_latex_error_table(
        act: int,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        method: str = DEFAULT_SIMULATION_METHOD,
        prefix: str = DEFAULT_DS_PREFIX,
        base_path: Path = DEFAULT_BASE_PATH,
) -> None:
    """
    Print a LaTeX table of ME / MSE per dataset for a single ACT value.

    Each cell shows ``ME / MSE`` with the best value per dataset in bold.
    Final columns show normalised rank sums across datasets.

    Args:
        act:       ACT value to process.
        ds_start:  First dataset index (inclusive).
        ds_end:    Last dataset index (inclusive).
        method:    Simulation method string.
        prefix:    Dataset file-name prefix.
        base_path: Root directory for accuracy data.
    """
    df = compute_all_errors(act, ds_start, ds_end, method, prefix, base_path)
    df = format_cells(df)
    df = compute_ranks(df)
    print(build_latex_error_table(df, summarize_ranks(df)))


def print_latex_rank_table(
        act_list: list[int] = DEFAULT_ACT_LIST,
        ds_start: int = DEFAULT_DS_START,
        ds_end: int = DEFAULT_DS_END,
        method: str = DEFAULT_SIMULATION_METHOD,
        prefix: str = DEFAULT_DS_PREFIX,
        base_path: Path = DEFAULT_BASE_PATH,
) -> None:
    """
    Print a LaTeX rank-summary table across multiple ACT values.

    Each column is one ACT value; each cell shows the normalised ME-rank
    sum for that estimator and ACT.  The best score per column is bolded.

    Args:
        act_list:  ACT values to include as columns.
        ds_start:  First dataset index (inclusive).
        ds_end:    Last dataset index (inclusive).
        method:    Simulation method string.
        prefix:    Dataset file-name prefix.
        base_path: Root directory for accuracy data.
    """
    all_summaries = []
    for act in act_list:
        df = compute_all_errors(act, ds_start, ds_end, method, prefix, base_path)
        df = compute_ranks(df)
        rs = summarize_ranks(df)
        rs["ACT"] = act
        all_summaries.append(rs)

    combined = pd.concat(all_summaries, ignore_index=True)
    table = (
        combined[["estimator", "ACT", "ME-rank-norm-latex"]]
        .pivot(index="estimator", columns="ACT", values="ME-rank-norm-latex")
        .reset_index()
    )
    table.columns = ["estimator"] + [f"ACT{act}" for act in act_list]
    print(table.to_latex(
        escape=False,
        column_format="c" * len(table.columns),
        index=False,
    ))


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == '__main__':
    # Comment in/out the outputs you want to produce.

    # Line plots of estimated ESS (one page per dataset, saved to PDF):
    plot_estimated_ess(act=5)
    # plot_estimated_ess(act=5, relative=True)

    # Box plot of relative mean error across ACT values:
    # plot_relative_error_boxplot()

    # LaTeX error table for one ACT value:
    # print_latex_error_table(act=5)

    # LaTeX rank summary table across all ACT values:
    # print_latex_rank_table()
    pass