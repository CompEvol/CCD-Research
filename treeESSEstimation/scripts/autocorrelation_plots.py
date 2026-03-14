"""
Autocorrelation Signature Plotting Tool
========================================

Reads autocorrelation data from CSV files and creates a 2×2 subplot grid
comparing three fixed simulation types (Simple, Noisy, RNNI) and a reference
MCMC run. Each panel shows one or more named autocorrelation curves.

Curves are processed before plotting:

1. Truncated at the first negative value (values below zero are not
   meaningful for ESS estimation with the Initial Positive Sequence Estimator).
2. A trailing zero is appended for visual consistency.
3. Normalised so the first value equals 1.0, making curves from different
   metrics directly comparable on the same axes.

File format
-----------
All files share the same plain-text format::

    <name> <value1> <value2> <value3> ...

One named curve per line, values separated by whitespace. Example::

    expRF 1.0 0.95 0.89 0.72 0.50 0.31 0.08
    logP  1.0 0.91 0.80 0.65 0.44 0.22 0.04

Expected file names (configurable via FILE_PATTERN / --file-pattern)::

    DS3-Simple-autocorrelations.csv
    DS3-Noisy-autocorrelations.csv
    DS3-RNNI-autocorrelations.csv
    DS3-Real-autocorrelations.csv      <- reference panel

Usage examples
--------------
Defaults (DS=DS3, all lines shown, saves plot)::

    python autocorrelation_plots.py

Override dataset and show only specific lines::

    python autocorrelation_plots.py --ds DS4 --display-lines expRF logP

Provide explicit file paths instead of relying on the naming pattern::

    python autocorrelation_plots.py \\
        --simple-file /data/run1-Simple.csv \\
        --reference-file /data/run1-Real.csv

Use a custom reference type name (changes the file looked up via pattern)::

    python autocorrelation_plots.py --reference-type MCMC
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# ==========================================================
# FILE CONFIGURATION  (edit here or override with CLI flags)
# ==========================================================

# Directory that contains the autocorrelation CSV files.
DEFAULT_BASE_PATH: Path = Path("..") / "example_data" / "autocorrelation"

# Dataset prefix used to construct file names (e.g. "DS3").
DEFAULT_DS: str = "DS3"

# File-name pattern.  Placeholders: {ds}, {type}.
FILE_PATTERN: str = "{ds}-{type}-autocorrelations.csv"

# Name used for the reference panel when building the file name from the
# pattern.  Change to match your actual file naming.
DEFAULT_REFERENCE_TYPE: str = "Real"

# ==========================================================
# DATA CONFIGURATION
# ==========================================================

# Which line names to display from the simulation files.
# Set to None to display every line found in the files.
DEFAULT_DISPLAY_LINES: list[str] | None = None  # e.g. ["expRF", "logP"]

# Which line names to display from the reference file.
# Set to None to display every line found.
DEFAULT_REFERENCE_DISPLAY: list[str] | None = None  # e.g. ["posterior", "kappa"]

# ==========================================================
# OUTPUT CONFIGURATION
# ==========================================================

DEFAULT_OUTPUT_DIR: Path = Path(__file__).parent  # or Path("..") / "plots"
DEFAULT_SAVE_PLOTS: bool = True

# ==========================================================
# VISUAL STYLE
# ==========================================================

PLOT_TITLE_FONT_SIZE: int = 14
AXES_FONT_SIZE: int = 14
LEGEND_FONT_SIZE: int = 14

# ==========================================================
# FIXED EXPERIMENT CONSTANTS
# ==========================================================

# The three simulation types are fixed for this experiment.
# The fourth 2×2 panel is always the reference MCMC run.
SIMULATION_TYPES: list[str] = ["Simple", "Noisy", "RNNI"]


# ==========================================================
# CORE PROCESSING
# ==========================================================

def tokenize_line(line: str) -> tuple[str | None, list[float]]:
    """
    Parse one line from an autocorrelation data file.

    Each line has a name token followed by whitespace-separated floats.

    Args:
        line: Raw text line, e.g. ``"expRF 1.0 0.95 0.89"``.

    Returns:
        A ``(name, values)`` tuple.  Returns ``(None, [])`` for blank lines.

    Example:
        >>> tokenize_line("expRF 1.0 0.95 0.89")
        ('expRF', [1.0, 0.95, 0.89])
    """
    tokens = line.strip().split()
    if not tokens:
        return None, []
    return tokens[0], [float(x) for x in tokens[1:]]


def process_curve(values: list[float]) -> list[float]:
    """
    Truncate and normalise a raw autocorrelation sequence for plotting.

    Processing steps:

    1. Truncate at the first negative value.
    2. Append a trailing zero so the plotted line descends to the axis.
    3. Normalise so the first value is 1.0.

    Args:
        values: Raw autocorrelation values.

    Returns:
        Processed values ready for plotting, or an empty list if *values*
        is empty.
    """
    if not values:
        return []

    truncated: list[float] = []
    for v in values:
        if v < 0:
            break
        truncated.append(v)

    truncated.append(0.0)

    first = truncated[0]
    if first == 0:
        return truncated
    return [v / first for v in truncated]


# ==========================================================
# I/O
# ==========================================================

def read_autocorrelation_file(
        file_path: Path,
        keep_lines: list[str] | None = None,
) -> dict[str, list[float]]:
    """
    Read one autocorrelation CSV file and return its named curves.

    Args:
        file_path:  Path to the CSV file.
        keep_lines: If given, only curves whose name appears in this list
                    are returned.  ``None`` returns all curves.

    Returns:
        A dict mapping curve name to its list of float values.

    Raises:
        SystemExit: If the file cannot be opened.
    """
    try:
        with open(file_path) as f:
            lines = f.readlines()
    except OSError as exc:
        print(f"Error: cannot read {file_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    curves: dict[str, list[float]] = {}
    for line in lines:
        name, values = tokenize_line(line)
        if name is None:
            continue
        if keep_lines is None or name in keep_lines:
            curves[name] = values
    return curves


def load_data(
        base_path: Path,
        ds: str,
        file_pattern: str,
        reference_type: str,
        sim_file_overrides: dict[str, Path],
        reference_file_override: Path | None,
        display_lines: list[str] | None,
        reference_display: list[str] | None,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[float]]]:
    """
    Read autocorrelation data for all simulation types and the reference run.

    Simulation files are looked up via *file_pattern* unless an explicit path
    is provided in *sim_file_overrides*.  The reference file follows the same
    pattern using *reference_type* as the type token, unless
    *reference_file_override* is set.

    Args:
        base_path:              Directory containing the CSV files.
        ds:                     Dataset prefix used in the file-name pattern.
        file_pattern:           Pattern with ``{ds}`` and ``{type}``
                                placeholders.
        reference_type:         Type token used for the reference file name.
        sim_file_overrides:     Explicit paths for individual simulation types,
                                keyed by type name. Missing entries fall back
                                to *file_pattern*.
        reference_file_override: Explicit path to the reference file,
                                 overriding *file_pattern*.
        display_lines:          Line names to keep from simulation files;
                                ``None`` keeps all.
        reference_display:      Line names to keep from the reference file;
                                ``None`` keeps all.

    Returns:
        A ``(sim_data, ref_data)`` tuple where *sim_data* maps each simulation
        type to its curve dict, and *ref_data* maps reference curve names to
        their value lists.
    """
    sim_data: dict[str, dict[str, list[float]]] = {}
    for sim_type in SIMULATION_TYPES:
        if sim_type in sim_file_overrides:
            path = sim_file_overrides[sim_type]
        else:
            filename = file_pattern.format(ds=ds, type=sim_type)
            path = base_path / filename
        print(f"  {sim_type}: {path}")
        sim_data[sim_type] = read_autocorrelation_file(path, display_lines)

    if reference_file_override is not None:
        ref_path = reference_file_override
    else:
        ref_filename = file_pattern.format(ds=ds, type=reference_type)
        ref_path = base_path / ref_filename
    print(f"  Reference: {ref_path}")
    ref_data = read_autocorrelation_file(ref_path, reference_display)

    return sim_data, ref_data


# ==========================================================
# PLOTTING
# ==========================================================

def _style_axis(ax, title: str) -> None:
    """
    Apply shared axis formatting: title, labels, x-origin, legend, grid.

    Args:
        ax:    Matplotlib [Axes] to style.
        title: Subplot title.
    """
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation (scaled)")
    xlim = list(ax.get_xlim())
    xlim[0] = 0
    ax.set_xlim(xlim)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _subplot_title(sim_type: str) -> str:
    """
    Return the display title for a simulation type panel.

    Args:
        sim_type: Simulation type name.

    Returns:
        Human-readable panel title.
    """
    if sim_type == "RNNI":
        return "RNNI Chain"
    return f"{sim_type} Repetition Chain"


def plot_simulation_subplot(
        ax,
        sim_type: str,
        curves: dict[str, list[float]],
) -> None:
    """
    Plot all curves for one simulation type.

    Each curve is processed with [process_curve] before plotting.

    Args:
        ax:       Matplotlib [Axes] to draw on.
        sim_type: Simulation type name, used for the panel title.
        curves:   Dict mapping curve names to raw value lists.
    """
    for name, values in curves.items():
        ax.plot(process_curve(values), label=name)
    _style_axis(ax, _subplot_title(sim_type))


def plot_reference_subplot(
        ax,
        curves: dict[str, list[float]],
) -> None:
    """
    Plot the reference MCMC run in the fourth panel.

    Each curve is processed with [process_curve] before plotting.
    Curves are coloured starting from position 2 in the ``tab10`` palette
    so they remain visually distinct from the simulation panels.

    Args:
        ax:     Matplotlib [Axes] to draw on.
        curves: Dict mapping curve names to raw value lists.
    """
    if not curves:
        ax.text(0.5, 0.5, "No reference data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Reference")
        return

    palette = plt.colormaps.get_cmap("tab10").colors
    for i, (name, values) in enumerate(curves.items()):
        color = palette[(i + 2) % len(palette)]
        ax.plot(process_curve(values), color=color, label=name, linewidth=2)

    _style_axis(ax, "Reference MCMC Chain")


def plot_signatures(
        sim_data: dict[str, dict[str, list[float]]],
        ref_data: dict[str, list[float]],
        title: str | None = None,
        output_path: Path | None = None,
) -> None:
    """
    Render the 2×2 autocorrelation signature grid.

    The first three panels correspond to [SIMULATION_TYPES] in order; the
    fourth shows the reference MCMC run.

    Args:
        sim_data:    Simulation curves keyed by type then curve name.
        ref_data:    Reference curves keyed by curve name.
        title:       Optional figure-level suptitle.
        output_path: If given, the figure is saved here in addition to
                     being shown interactively.
    """
    plt.rcParams["font.size"] = PLOT_TITLE_FONT_SIZE
    plt.rcParams["axes.titlesize"] = AXES_FONT_SIZE
    plt.rcParams["axes.labelsize"] = AXES_FONT_SIZE
    plt.rcParams["xtick.labelsize"] = AXES_FONT_SIZE
    plt.rcParams["ytick.labelsize"] = AXES_FONT_SIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONT_SIZE

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, sim_type in enumerate(SIMULATION_TYPES):
        plot_simulation_subplot(axes[idx], sim_type, sim_data[sim_type])
    plot_reference_subplot(axes[3], ref_data)

    if title:
        fig.suptitle(title, fontsize=PLOT_TITLE_FONT_SIZE)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        print(f"  Saved: {output_path}")

    plt.show()


# ==========================================================
# CLI
# ==========================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Populated [argparse.Namespace].
    """
    parser = argparse.ArgumentParser(
        description="Plot 2×2 autocorrelation signatures for a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument(
        "--base-path", type=Path, default=DEFAULT_BASE_PATH,
        help="Directory containing the autocorrelation CSV files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory where plots are saved.",
    )

    # Dataset
    parser.add_argument(
        "--ds", default=DEFAULT_DS,
        help="Dataset prefix used in file names, e.g. DS7.",
    )

    # File naming
    parser.add_argument(
        "--file-pattern", default=FILE_PATTERN,
        help="File-name pattern.  Placeholders: {ds}, {type}.",
    )
    parser.add_argument(
        "--reference-type", default=DEFAULT_REFERENCE_TYPE,
        help="Type token for the reference file in the name pattern.",
    )
    parser.add_argument(
        "--simple-file", type=Path, default=None,
        help="Explicit path to the Simple simulation file (overrides pattern).",
    )
    parser.add_argument(
        "--noisy-file", type=Path, default=None,
        help="Explicit path to the Noisy simulation file (overrides pattern).",
    )
    parser.add_argument(
        "--rnni-file", type=Path, default=None,
        help="Explicit path to the RNNI simulation file (overrides pattern).",
    )
    parser.add_argument(
        "--reference-file", type=Path, default=None,
        help="Explicit path to the reference file (overrides pattern).",
    )

    # Line selection
    parser.add_argument(
        "--display-lines", nargs="+", default=DEFAULT_DISPLAY_LINES,
        metavar="NAME",
        help="Line names to show from simulation files.  Omit to show all.",
    )
    parser.add_argument(
        "--reference-display", nargs="+", default=DEFAULT_REFERENCE_DISPLAY,
        metavar="NAME",
        help="Line names to show from the reference file.  Omit to show all.",
    )

    # Output
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save", dest="save", action="store_true", default=DEFAULT_SAVE_PLOTS,
        help="Save the plot to --output-dir.",
    )
    save_group.add_argument(
        "--no-save", dest="save", action="store_false",
        help="Do not save the plot.",
    )

    return parser.parse_args()


# ==========================================================
# ENTRY POINT
# ==========================================================

def main() -> None:
    """Parse arguments, load data, and render the 2×2 plot."""
    args = parse_arguments()

    sim_file_overrides: dict[str, Path] = {}
    if args.simple_file:
        sim_file_overrides["Simple"] = args.simple_file
    if args.noisy_file:
        sim_file_overrides["Noisy"] = args.noisy_file
    if args.rnni_file:
        sim_file_overrides["RNNI"] = args.rnni_file

    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data for {args.ds} ...")
    sim_data, ref_data = load_data(
        base_path=args.base_path,
        ds=args.ds,
        file_pattern=args.file_pattern,
        reference_type=args.reference_type,
        sim_file_overrides=sim_file_overrides,
        reference_file_override=args.reference_file,
        display_lines=args.display_lines,
        reference_display=args.reference_display,
    )

    output_path = None
    if args.save:
        output_path = args.output_dir / f"{args.ds}-signatures.pdf"

    print("Plotting ...")
    plot_signatures(
        sim_data=sim_data,
        ref_data=ref_data,
        title=args.ds,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()