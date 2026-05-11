"""Plot difference heatmaps between chess_champion_distances and UMAP test distances.

This script loads:
 - `data/processed/chess_champion_distances.parquet` (the 'chess' reference)
 - `results/evaluation/distances_test_umap.parquet` (the UMAP test distances)

It constructs ordered square matrices for both, aligns them on the same player order,
computes difference matrices and saves heatmaps.

Added: a normalized difference heatmap (min-max across each matrix before diff).

Outputs:
 - results/graphics/jsd_heatmap_diff_signed_chess_minus_umap.pdf
 - results/graphics/jsd_heatmap_diff_abs_chess_vs_umap.pdf
 - results/graphics/jsd_heatmap_diff_signed_norm_chess_minus_umap.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

# Styling constants
FIG_WIDTH = 3.5
HEATMAP_HEIGHT = 4.5
HEATMAP_CMAP = "magma_r"

# Input paths
CHESS_INPUT = Path("data/processed/chess_champion_distances.parquet")
UMAP_DISTANCES = Path("results/evaluation/distances_test_umap.parquet")

# Output paths
OUT_SIGNED = Path("results/graphics/diff_jsd_signed.pdf")
OUT_ABS = Path("results/graphics/diff_jsd_abs.pdf")
OUT_NORM_SIGNED = Path("results/graphics/diff_jsd_signed_norm.pdf")
OUT_NORM_ABS = Path("results/graphics/diff_jsd_abs_norm.pdf")
OUT_SIGNED.parent.mkdir(parents=True, exist_ok=True)

# Seaborn theme matching project
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


def _to_square_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame (long or wide) into a square pandas DataFrame.

    Supported input forms:
    - long: columns containing 'p1', 'p2', 'distance' (case-insensitive)
    - wide: a column with player names used as index, or columns as player names
    """
    cols_lower = [c.lower() for c in df.columns]
    # long format detection
    if set(["p1", "p2", "distance"]).issubset(set(cols_lower)):
        col_map = {c.lower(): c for c in df.columns}
        p1 = col_map["p1"]
        p2 = col_map["p2"]
        val = col_map["distance"]

        pdf = df.to_pandas()
        players = sorted(
            set(pdf[p1].unique().tolist()) | set(pdf[p2].unique().tolist())
        )

        mirrored = pdf[[p1, p2, val]].rename(columns={p1: "p1", p2: "p2", val: "value"})
        mirrored_rev = mirrored.rename(columns={"p1": "p2", "p2": "p1"})
        diag = pd.DataFrame(
            {"p1": players, "p2": players, "value": [0.0] * len(players)}
        )
        combined = pd.concat([mirrored, mirrored_rev, diag], ignore_index=True)
        combined = combined.drop_duplicates(subset=["p1", "p2"])  # keep first
        matrix = combined.pivot(index="p1", columns="p2", values="value")
        matrix = matrix.reindex(index=players, columns=players)
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        return matrix

    # wide format heuristics
    pdf = df.to_pandas()

    def _strip_quotes(x):
        s = str(x)
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            return s[1:-1]
        return s

    # If there is a candidate column to use as index (dtype object and unique == rows)
    candidate_index_cols = [
        c
        for c in pdf.columns
        if pdf[c].dtype == object and pdf[c].nunique() == pdf.shape[0]
    ]
    if candidate_index_cols:
        idx_col = candidate_index_cols[0]
        matrix_pd = pdf.set_index(idx_col)
    else:
        matrix_pd = pdf.copy()

    # Clean index/columns quotes conservatively
    try:
        matrix_pd.columns = [_strip_quotes(c) for c in matrix_pd.columns]
    except Exception:
        pass
    try:
        matrix_pd.index = [_strip_quotes(i) for i in matrix_pd.index]
    except Exception:
        pass

    # If the index looks numeric (index lost) but columns are names, set index to columns
    try:
        idx_is_numeric_like = all(str(i).isdigit() for i in matrix_pd.index)
    except Exception:
        idx_is_numeric_like = False
    try:
        cols_are_non_numeric = not all(str(c).isdigit() for c in matrix_pd.columns)
    except Exception:
        cols_are_non_numeric = True
    if (
        idx_is_numeric_like
        and matrix_pd.shape[0] == matrix_pd.shape[1]
        and cols_are_non_numeric
    ):
        matrix_pd.index = matrix_pd.columns
        matrix_pd = matrix_pd.apply(pd.to_numeric, errors="coerce")
        return matrix_pd

    # If square already or transpose makes it square, handle those cases
    if matrix_pd.shape[0] == matrix_pd.shape[1]:
        matrix_pd = matrix_pd.apply(pd.to_numeric, errors="coerce")
        return matrix_pd

    t = matrix_pd.T.copy()
    if t.shape[0] == t.shape[1]:
        t = t.apply(pd.to_numeric, errors="coerce")
        return t

    # If number of rows equals number of columns but index missing (common when index lost), set index to column names
    if matrix_pd.shape[0] == len(matrix_pd.columns):
        matrix_pd.index = matrix_pd.columns
        matrix_pd = matrix_pd.apply(pd.to_numeric, errors="coerce")
        return matrix_pd

    # Heuristic: look for a column whose cleaned values equal the set of column names
    col_names_set = set([_strip_quotes(c) for c in matrix_pd.columns])
    for c in matrix_pd.columns:
        if (
            matrix_pd[c].dtype == object
            and matrix_pd[c].nunique() == matrix_pd.shape[0]
        ):
            vals = [_strip_quotes(v) for v in matrix_pd[c].astype(str).tolist()]
            if set(vals) == col_names_set:
                matrix_pd.index = vals
                try:
                    matrix_pd = matrix_pd.drop(columns=[c])
                except Exception:
                    pass
                matrix_pd = matrix_pd.apply(pd.to_numeric, errors="coerce")
                return matrix_pd

    # Fallback: coerce to numeric and return (caller will handle NaNs after reindexing)
    numeric = matrix_pd.apply(pd.to_numeric, errors="coerce")
    return numeric


# Read inputs
ch_df = pl.read_parquet(str(CHESS_INPUT))
um_df = pl.read_parquet(str(UMAP_DISTANCES))

# Convert to square matrices
ch_mat = _to_square_matrix(ch_df)
um_mat = _to_square_matrix(um_df)

# Normalize labels (strip surrounding quotes and whitespace only)
ch_mat.index = ch_mat.index.astype(str).str.strip().str.strip('"').str.strip("'")
ch_mat.columns = ch_mat.columns.astype(str).str.strip().str.strip('"').str.strip("'")
um_mat.index = um_mat.index.astype(str).str.strip().str.strip('"').str.strip("'")
um_mat.columns = um_mat.columns.astype(str).str.strip().str.strip('"').str.strip("'")

# Unified player order: preserve chess order, then add umap-only players
ch_players = list(ch_mat.index)
um_players = list(um_mat.index)
players = ch_players + [p for p in um_players if p not in set(ch_players)]

# Reindex and coerce numeric
ch_values = ch_mat.reindex(index=players, columns=players).apply(
    pd.to_numeric, errors="coerce"
)
um_values = um_mat.reindex(index=players, columns=players).apply(
    pd.to_numeric, errors="coerce"
)

# Compute diffs
diff_signed = ch_values - um_values
diff_abs = diff_signed.abs()


# Normalization: min-max per matrix (ignoring NaNs)
def _min_max_normalize(mat: pd.DataFrame) -> pd.DataFrame:
    mat_numeric = mat.apply(pd.to_numeric, errors="coerce")
    minv = mat_numeric.min().min()
    maxv = mat_numeric.max().max()
    if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
        return mat_numeric.copy()
    return (mat_numeric - minv) / (maxv - minv)


ch_norm = _min_max_normalize(ch_values)
um_norm = _min_max_normalize(um_values)

# Diff on normalized matrices
diff_signed_norm = ch_norm - um_norm
diff_abs_norm = diff_signed_norm.abs()

# Plot helper


def _save_heatmap(
    matrix, out_path, cmap, fmt=".2f", center=None, annot=True, title=None
):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, HEATMAP_HEIGHT))

    if matrix.isnull().all().all():
        ax.text(0.5, 0.5, "All values are NaN", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(str(out_path), format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)
        return

    mask = matrix.isnull()

    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        annot_kws={"size": 2.2},
        square=True,
        linewidths=0.01,
        linecolor="#CCCCCC",
        cbar_kws={
            "label": "Jensen-Shannon Divergence difference",
            "shrink": 0.5,
            "pad": 0.04,
        },
        center=center,
        mask=mask,
        xticklabels=[str(x) for x in matrix.columns],
        yticklabels=[str(y) for y in matrix.index],
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(
        [str(x) for x in matrix.columns], rotation=90, ha="center", fontsize=3.5
    )
    ax.set_yticklabels([str(y) for y in matrix.index], rotation=0, fontsize=3.5)
    if title:
        plt.title(title)
    plt.tight_layout()
    fig.savefig(str(out_path), format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


# Signed diff: diverging colormap centered at zero
_save_heatmap(
    diff_signed,
    OUT_SIGNED,
    cmap=HEATMAP_CMAP,
    fmt=".2f",
    center=0.0,
    annot=True,
    title="Signed diff",
)

# Absolute diff: project style
_save_heatmap(
    diff_abs,
    OUT_ABS,
    cmap=HEATMAP_CMAP,
    fmt=".3f",
    center=None,
    annot=True,
    title="Absolute diff",
)

# Normalized signed diff
_save_heatmap(
    diff_signed_norm,
    OUT_NORM_SIGNED,
    cmap=HEATMAP_CMAP,
    fmt=".2f",
    center=0.0,
    annot=True,
    title="Normalized signed diff",
)

# Normalized absolute diff
_save_heatmap(
    diff_abs_norm,
    OUT_NORM_ABS,
    cmap=HEATMAP_CMAP,
    fmt=".3f",
    center=None,
    annot=True,
    title="Normalized absolute diff",
)

print(f"Saved signed diff heatmap to {OUT_SIGNED}")
print(f"Saved absolute diff heatmap to {OUT_ABS}")
print(f"Saved normalized signed diff heatmap to {OUT_NORM_SIGNED}")
print(f"Saved normalized absolute diff heatmap to {OUT_NORM_ABS}")
