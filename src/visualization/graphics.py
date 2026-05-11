"""
Refactored visualization utilities for the ML_Dead_Chess_Champions project.

This module provides routines to produce publication-quality figures
for:
 - Jensen-Shannon Divergence heatmaps (symmetric, test-only)
 - Asymmetric stability heatmaps (train vs test)
 - Moves-per-player distribution bar chart

Design notes
------------
- All plotting routines accept a `Config` object which encapsulates file paths and
  configuration parameters. This keeps plotting code pure and free of global state.
- I/O is robust: parquet read errors and missing files are handled gracefully with
  informative logging. Figure saving ensures parent directories exist.
- Helper functions centralize common operations (DataFrame loading, matrix creation,
  figure saving) so that the plotting functions focus on visual logic.
- Type hints and concise docstrings are provided for better maintainability.

Author: refactor by assistant
"""

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

from src.core.config import Config
from src.core.utils import getLogger

# Module-level logger
logger = getLogger()

# Seaborn theme configuration chosen for clarity and reproducibility in academic figures.
sns.set_theme(
    context="paper",
    style="ticks",
    font="STIXGeneral",
    palette="colorblind",
    rc={
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    },
)

# Figure size constants in inches
FIG_WIDTH = 3.45
FIG_WIDTH_DOUBLE = 7.25
HEATMAP_HEIGHT = 4.5
DISTRIBUTION_HEIGHT = 2.8

# Color map names used consistently across figures
HEATMAP_CMAP = "magma_r"  # reversed for heatmaps: bright = similar (low distance)
BAR_CMAP = "magma"  # forward magma for bar color mapping


# ---------- Helper utilities ----------


def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of `path` exists. Creates it if necessary."""
    p = Path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory: %s", str(p.parent))


def _safe_read_parquet(path: str) -> Optional[pl.DataFrame]:
    """
    Read a parquet file into a Polars DataFrame with robust error handling.

    Returns None and logs a descriptive error if reading fails.
    """
    try:
        df = pl.read_parquet(path)
        logger.debug("Loaded parquet file: %s (rows=%d)", path, df.height)
        return df
    except FileNotFoundError:
        logger.error("Parquet file not found: %s", path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to read parquet %s: %s", path, exc)
    return None


def _save_figure(
    fig: Figure, out_path: str, *, fmt: str = "pdf", dpi: int = 600
) -> None:
    """
    Save a Matplotlib figure to disk ensuring the output directory exists.

    The function logs success and handles filesystem errors gracefully.
    """
    _ensure_parent_dir(out_path)
    try:
        fig.savefig(out_path, format=fmt, dpi=dpi, bbox_inches="tight")
        logger.info("Saved figure to %s", out_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Unable to save figure to %s: %s", out_path, exc)


def _to_ordered_square_matrix(
    df: pl.DataFrame,
    p1_col: str,
    p2_col: str,
    value_col: str,
    ordered_players: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Convert a long-format Polars DataFrame into a pandas square matrix (DataFrame) indexed
    and columned by player names in `ordered_players`.

    If `ordered_players` is None the function determines the sorted union of players
    occurring in `p1_col` and `p2_col`.
    """
    # Compute players union in a stable sorted order if not provided
    if ordered_players is None:
        p1_names = df.select(pl.col(p1_col)).unique().to_series().to_list()
        p2_names = df.select(pl.col(p2_col)).unique().to_series().to_list()
        players = sorted(set(p1_names) | set(p2_names))
    else:
        players = list(ordered_players)

    # Create mirror and diagonal entries to ensure a full square matrix
    mirror = df.select(
        [pl.col(p2_col).alias(p1_col), pl.col(p1_col).alias(p2_col), pl.col(value_col)]
    )
    diag = pl.DataFrame(
        {p1_col: players, p2_col: players, value_col: [0.0] * len(players)}
    )

    combined = pl.concat([df, mirror, diag]).unique(subset=[p1_col, p2_col])
    # Pivot to pandas to leverage seaborn's heatmap which works naturally with pandas
    matrix_pd = (
        combined.to_pandas()
        .pivot(index=p1_col, columns=p2_col, values=value_col)
        .reindex(index=players, columns=players)
    )
    return matrix_pd


# ---------- Plotting routines ----------


def _choose_annotation_settings(
    n: int, double_column: bool, default_fmt: str = ".2f"
) -> dict:
    """
    Return annotation settings tuned for visibility. The user requires values, so
    annotations are enabled for all matrix sizes; font size and formatting are
    adapted to keep annotations as legible as possible.

    Parameters
    - n: matrix dimension (rows)
    - double_column: whether the figure spans two columns (larger fonts)
    - default_fmt: numeric format string passed to seaborn (e.g. '.2f' or '.4f')
    """
    # Font size heuristics (bigger for double-column layouts)
    if n <= 8:
        size = 9 if double_column else 7
    elif n <= 15:
        size = 7 if double_column else 5
    elif n <= 25:
        size = 5 if double_column else 4
    else:
        # Very large matrices: still annotate but keep font small to avoid overlap
        size = 4 if double_column else 3

    return {"annot": True, "fmt": default_fmt, "annot_kws": {"size": size}}


def _render_heatmap(
    matrix_pd: pd.DataFrame,
    out_path: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    double_column: bool = True,
    cmap: str = HEATMAP_CMAP,
    cbar_label: str = "Jensen-Shannon Divergence",
    annot_fmt: str = ".2f",
) -> None:
    """
    Common heatmap rendering routine that centralizes figure sizing, annotation rules
    and colorbar placement to produce publication-ready output for single/double
    column layouts.
    """
    # Choose width for single or double column
    width = FIG_WIDTH_DOUBLE if double_column else FIG_WIDTH
    fig, ax = plt.subplots(figsize=(width, HEATMAP_HEIGHT))

    n = 0
    try:
        n = matrix_pd.shape[0]
    except Exception:
        n = 0

    annot_settings = _choose_annotation_settings(
        n, double_column, default_fmt=annot_fmt
    )

    # Draw heatmap with square cells (square=True) for consistent aspect
    mesh = sns.heatmap(
        matrix_pd,
        ax=ax,
        cmap=cmap,
        square=True,
        linewidths=0.4 if n <= 30 else 0.15,
        linecolor="#DDDDDD",
        cbar_kws={"label": cbar_label, "fraction": 0.05, "pad": 0.03},
        vmin=0.0,
        vmax=1.0,
        **annot_settings,
    )

    # Always improve annotation visibility: per-cell contrasting text color, a semi-opaque bbox
    # behind each label and a strong stroke outline. This aims for maximal legibility in print.
    # safe-get the matrix values as a numpy array; handle non-numeric gracefully
    try:
        values = matrix_pd.values
    except Exception:
        values = None

    # Build colormap and normalizer consistent with the heatmap call
    cmap_obj = cm.get_cmap(cmap)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Use matplotlib path effects to add a stroke outline to text for extra contrast
    try:
        import matplotlib.patheffects as patheffects  # local import to avoid global overhead
    except Exception:
        patheffects = None

    texts = ax.texts  # list of Text objects created by seaborn when annot=True
    if values is not None and len(texts) == values.size:
        # Ensure row-major order mapping between texts and values
        rows, cols = values.shape
        k = 0
        for i in range(rows):
            for j in range(cols):
                txt = texts[k]
                k += 1
                try:
                    v = float(values[i, j])
                except Exception:
                    # Skip non-numeric (e.g., NaN) cells
                    continue

                rgba = cmap_obj(norm(v))

                # Convert sRGB to linear RGB for luminance per WCAG
                def to_linear(c: float) -> float:
                    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

                r_lin = to_linear(rgba[0])
                g_lin = to_linear(rgba[1])
                b_lin = to_linear(rgba[2])
                lum = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

                # Contrast ratio with white and black according to WCAG
                def contrast_ratio(l1: float, l2: float) -> float:
                    lighter = max(l1, l2)
                    darker = min(l1, l2)
                    return (lighter + 0.05) / (darker + 0.05)

                cr_white = contrast_ratio(1.0, lum)
                cr_black = contrast_ratio(0.0, lum)

                # Choose the text color that yields better contrast
                if cr_white >= cr_black:
                    text_color = "white"
                    outline_color = "black"
                else:
                    text_color = "black"
                    outline_color = "white"

                # Apply color to text
                txt.set_color(text_color)

                # Apply a semi-opaque bbox behind text to maximize readability
                # Use the outline_color as bbox facecolor (contrasting) with alpha
                try:
                    bbox_face = outline_color
                    bbox_props = dict(
                        boxstyle="round,pad=0.18",
                        linewidth=0,
                        facecolor=bbox_face,
                        alpha=0.68,
                    )
                    txt.set_bbox(bbox_props)
                except Exception:
                    pass

                # apply a stronger stroke outline for readability when available
                if patheffects is not None:
                    # Slightly stronger stroke for readability
                    stroke_w = 1.8 if (n <= 12) else 1.2
                    txt.set_path_effects(
                        [
                            patheffects.Stroke(
                                linewidth=stroke_w, foreground=outline_color
                            ),
                            patheffects.Normal(),
                        ]
                    )

    # Axis labels
    ax.set_title(title, fontsize=10 if double_column else 9)
    ax.set_xlabel(xlabel, fontsize=8 if double_column else 7)
    ax.set_ylabel(ylabel, fontsize=8 if double_column else 7)

    # Tick label sizing and rotation
    xt_fs = 7 if double_column else 4.5
    yt_fs = 7 if double_column else 4.5
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", fontsize=xt_fs)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=yt_fs)

    # Improve layout and save
    sns.despine(ax=ax)
    fig.tight_layout()

    _save_figure(fig, out_path)
    plt.close(fig)


def jsd_heatmap(config: Config, *, double_column: bool = True) -> None:
    """
    Generate a symmetric Jensen-Shannon Divergence heatmap (test set only) and save it as a PDF.

    Visualization choices:
    - Colormap: reversed 'magma' so low distances (similarity) appear bright and
      large distances appear dark.
    - For larger player matrices annotations are disabled to avoid clutter.
    - The caller can request `double_column=True` to produce a wider figure
    """
    # Load precomputed distances for the test set
    distances_path = config.paths.get_distances_path(
        method=config.jsd.method, is_test=True, kde=config.jsd.kde
    )
    df = _safe_read_parquet(distances_path)
    if df is None:
        logger.error(
            "JSD heatmap aborted: could not load distances from %s", distances_path
        )
        return

    # If the distances parquet exists but is empty, skip plotting
    try:
        if df.height == 0:
            logger.warning(
                "JSD heatmap aborted: distances file %s is empty. Run evaluation first.",
                distances_path,
            )
            return
    except Exception:
        # Defensive: if df doesn't implement height, convert to pandas
        if df.to_pandas().empty:
            logger.warning(
                "JSD heatmap aborted: distances file %s appears empty. Run evaluation first.",
                distances_path,
            )
            return

    # Build the full square matrix (symmetric) for plotting
    matrix_df = _to_ordered_square_matrix(
        df, p1_col="p1", p2_col="p2", value_col="distance"
    )

    # Render with the centralized heatmap routine
    out_path = config.paths.jsd_heatmap_path
    # title = f"Jensen-Shannon Divergence ({config.jsd.method})"
    _render_heatmap(
        matrix_df,
        out_path,
        # title=title,
        xlabel="",
        ylabel="",
        double_column=double_column,
        cmap=HEATMAP_CMAP,
    )


def moves_distribution(config: Config, top_n: Optional[int] = None) -> None:
    """
    Generate a bar chart showing the number of moves per player.

    Parameters
    ----------
    config : Config
        Project configuration object containing dataset path and output locations.
    top_n : Optional[int]
        If provided, restricts the visualization to the top_n players by move count.
    """
    df = _safe_read_parquet(config.paths.dataset_path)
    if df is None:
        logger.error(
            "Moves distribution aborted: dataset file not found at %s",
            config.paths.dataset_path,
        )
        return

    # Compute counts using Polars (fast for large datasets)
    counts = df.group_by("player_name").len().sort("len", descending=True)
    if counts.height == 0:
        logger.warning("Moves distribution: no players found in dataset.")
        return

    if top_n is not None:
        counts = counts.head(top_n)

    pdf = counts.to_pandas()

    # Create a simple bar chart using Matplotlib and map colors with magma colormap
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, DISTRIBUTION_HEIGHT))

    # Data for plotting
    players = pdf["player_name"].astype(str).tolist()
    heights = pdf["len"].astype(float).tolist()
    x = range(len(players))

    # Normalize heights to [0,1] for colormap mapping
    if heights:
        vmin, vmax = min(heights), max(heights)
    else:
        vmin, vmax = 0.0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(BAR_CMAP)

    # Map each bar height to an RGBA color
    bar_colors = [cmap(norm(h)) for h in heights]

    # Draw bars
    ax.bar(x, heights, color=bar_colors, edgecolor="#222222", linewidth=0.5)

    # Set tick labels centered under each bar
    ax.set_xticks(list(x))
    ax.set_xticklabels(players, rotation=90, ha="center", fontsize=6)

    ax.set_title("")  # Title omitted; prefer captioning in LaTeX
    ax.set_xlabel("Player", fontsize=8)
    ax.set_ylabel("Number of Moves", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=7)

    # Subtle grid behind bars for interpretability
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine()

    _save_figure(fig, config.paths.moves_distribution_graph_path)
    plt.close(fig)


def stability_heatmap(config: Config, *, double_column: bool = True) -> None:
    """
    Generate an asymmetric 'stability' heatmap comparing train-set players (rows)
    to test-set players (columns). The diagonal illustrates within-player similarity.

    The caller can set `double_column=True` to produce a wider figure
    """
    method = config.jsd.method
    kde = config.jsd.kde

    input_path = config.paths.get_full_cross_matrix_path(method, kde)
    df = _safe_read_parquet(input_path)
    if df is None:
        logger.error(
            "Stability heatmap aborted: input file missing (%s). Please run evaluation first.",
            input_path,
        )
        return

    # Determine player ordering from training player column to keep the diagonal aligned.
    try:
        players_ordered = (
            df.select("p_train").unique().sort("p_train").to_series().to_list()
        )
    except Exception:  # pragma: no cover - defensive fallback
        players_ordered = None

    # Pivot to a rectangular matrix. Use pandas pivot and then reindex to maintain order.
    matrix_pd = (
        df.to_pandas()
        .pivot(index="p_train", columns="p_test", values="distance")
        .reindex(index=players_ordered, columns=players_ordered)
    )

    out_path = config.paths.jsd_stability_heatmap_path
    # title = f"Stability (Train vs Test) — {method}"
    _render_heatmap(
        matrix_pd,
        out_path,
        # title=title,
        xlabel="Test Set Players",
        ylabel="Train Set Players",
        double_column=double_column,
        cmap=HEATMAP_CMAP,
        annot_fmt=".3f",
    )


def learning_curves(config: Config) -> None:
    """
    Generate learning curves (loss and accuracy) from the training history parquet file.

    The resulting figure contains two subplots:
    1. Cross-Entropy Loss on the training set.
    2. Predictive Accuracy on both the training and test sets.
    """
    history_path = config.paths.learning_curves_path
    df = _safe_read_parquet(str(history_path))

    if df is None:
        logger.error(
            "Learning curves aborted: missing %s. Run training first.", history_path
        )
        return

    pdf = df.to_pandas()
    epochs = pdf["epoch"]

    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH * 2.2, DISTRIBUTION_HEIGHT))

    # --- Subplot 1 : Loss ---
    ax1.plot(
        epochs,
        pdf["train_loss"],
        label="Train Loss",
        color="crimson",
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=False)

    # --- Subplot 2 : Accuracy ---
    ax2.plot(
        epochs,
        pdf["train_accuracy"],
        label="Train Accuracy",
        color="navy",
        marker="o",
        markersize=3,
        linewidth=1.5,
    )
    ax2.plot(
        epochs,
        pdf["test_accuracy"],
        label="Test Accuracy",
        color="forestgreen",
        marker="s",
        markersize=3,
        linewidth=1.5,
    )
    ax2.set_title("Accuracy (Train vs Test)")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=False)

    sns.despine(fig=fig)
    fig.tight_layout()

    out_path = str(Path(config.paths.result) / "graphics" / "learning_curves.pdf")
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_all_graphics(config: Config) -> None:
    """
    Convenience function to generate all standard graphics in sequence.

    Each plotting function logs its own progress and failures, allowing this routine
    to run unattended in batch evaluation workflows.
    """
    logger.info("Generating learning curves graph...")
    learning_curves(config)

    logger.info("Generating moves distribution graph...")
    moves_distribution(config)

    # Generate method-specific JSD and stability heatmaps (uses templates in config.paths)
    logger.info("Generating JSD and stability heatmaps for configured methods...")
    generate_model_graphics(config)


def generate_model_graphics(
    config: Config, methods: list | None = None, *, double_column: bool = True
) -> None:
    """
    Generate JSD heatmaps and stability heatmaps for a list of embedding methods.

    Parameters
    ----------
    config : Config
        Project configuration object. The function temporarily adjusts output
        paths so that figures for each method are written to distinct files
        named with the method suffix (e.g. `jsd_heatmap_maia2.pdf`).
    methods : list | None
        If omitted the function will default to a reasonable set covering both
        the installed pipeline (`umap`) and the Maia variants we produce
        (`maia2`, `maia2_ft`, `maia2_ft_mcts`). You can pass any method name that
        matches the naming convention used in the `data/processed` parquet files
        (train_{method}.parquet / test_{method}.parquet).
    """
    if methods is None:
        methods = ["umap", "maia2", "maia2_ft", "maia2_ft_mcts"]

    # Preserve originals
    orig_jsd_path = config.paths.jsd_heatmap_path
    orig_stab_path = config.paths.jsd_stability_heatmap_path
    orig_method = config.jsd.method

    for method in methods:
        logger.info("Generating graphics for method: %s", method)

        # Adjust config to point to this method and set method-specific output paths from templates
        config.jsd.method = method
        jsd_out = config.paths.method_jsd_heatmap_template.format(method=method)
        stab_out = config.paths.method_jsd_stability_template.format(method=method)
        stab_realpred_out = config.paths.method_jsd_stability_real_pred_template.format(
            method=method
        )

        # Temporarily override the output paths used by plotting functions
        config.paths.jsd_heatmap_path = jsd_out
        config.paths.jsd_stability_heatmap_path = stab_out

        try:
            jsd_heatmap(config, double_column=double_column)
        except Exception as exc:
            logger.error("Failed to generate JSD heatmap for %s: %s", method, exc)

        try:
            stability_heatmap(config, double_column=double_column)
        except Exception as exc:
            logger.error("Failed to generate stability heatmap for %s: %s", method, exc)

    # Restore originals
    config.paths.jsd_heatmap_path = orig_jsd_path
    config.paths.jsd_stability_heatmap_path = orig_stab_path
    config.jsd.method = orig_method
