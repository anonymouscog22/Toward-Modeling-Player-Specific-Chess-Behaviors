"""Plot heatmap from data/processed/chess_champion_distances.parquet

This script reads the same parquet as `test.py` and reproduces the exact style
used by the project's `jsd_heatmap` plotting routine (colors, fonts, sizes,
annotation formatting, figure dimensions and colorbar settings).

Output:
    results/graphics/jsd_heatmap_exact_style.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib import cm

# Styling constants copied from src/visualization/graphics.py
FIG_WIDTH = 3.5
HEATMAP_HEIGHT = 4.5
HEATMAP_CMAP = "magma_r"

INPUT = Path("data/processed/chess_champion_distances.parquet")
OUT = Path("results/graphics/jsd_heatmap_playstyle.pdf")

OUT.parent.mkdir(parents=True, exist_ok=True)

# Seaborn theme configuration matching the project's settings
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

# Read parquet using Polars (same as test.py)
df = pl.read_parquet(str(INPUT))


def _to_square_matrix(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame in either long (p1,p2,distance) or wide form
    into a square pandas DataFrame indexed/columned by player names.
    """
    cols = [c.lower() for c in df.columns]
    if set(["p1", "p2", "distance"]).issubset(set(cols)):
        # long-format
        col_map = {c.lower(): c for c in df.columns}
        p1_col = col_map["p1"]
        p2_col = col_map["p2"]
        value_col = col_map["distance"]

        p1_names = df.select(pl.col(p1_col)).unique().to_series().to_list()
        p2_names = df.select(pl.col(p2_col)).unique().to_series().to_list()
        players = sorted(set(p1_names) | set(p2_names))

        mirror = df.select(
            [
                pl.col(p2_col).alias(p1_col),
                pl.col(p1_col).alias(p2_col),
                pl.col(value_col),
            ]
        )
        diag = pl.DataFrame(
            {p1_col: players, p2_col: players, value_col: [0.0] * len(players)}
        )
        combined = pl.concat([df, mirror, diag]).unique(subset=[p1_col, p2_col])
        matrix_pd = (
            combined.to_pandas()
            .pivot(index=p1_col, columns=p2_col, values=value_col)
            .reindex(index=players, columns=players)
        )
        return matrix_pd

    # wide-format: use heuristics
    pdf = df.to_pandas()

    # If there is a column that contains unique strings equal to number of rows, use it as index
    candidate_index_cols = [
        c
        for c in pdf.columns
        if pdf[c].dtype == object and pdf[c].nunique() == pdf.shape[0]
    ]
    if candidate_index_cols:
        # pick the first candidate
        idx_col = candidate_index_cols[0]
        matrix_pd = pdf.set_index(idx_col)
    else:
        matrix_pd = pdf.copy()

    # If still not square, try transpose or set index to column names when appropriate
    if matrix_pd.shape[0] != matrix_pd.shape[1]:
        t = matrix_pd.T
        if t.shape[0] == t.shape[1]:
            matrix_pd = t
        elif matrix_pd.shape[0] == matrix_pd.shape[1] == len(matrix_pd.columns):
            # nothing
            pass
        else:
            # last resort: if columns are player-like names and length matches rows, set index to column names
            if matrix_pd.shape[0] == len(matrix_pd.columns):
                matrix_pd.index = matrix_pd.columns

    return matrix_pd


matrix_pd = _to_square_matrix(df)

# Ensure numeric values where appropriate
matrix_pd = matrix_pd.apply(pd.to_numeric, errors="coerce")

# Clean tick labels: remove extraneous quotes if present
matrix_pd.index = matrix_pd.index.astype(str).str.replace('"', "")
matrix_pd.columns = matrix_pd.columns.astype(str).str.replace('"', "")

# If index is a simple range (0..n-1) but columns contain player names and are same length,
# use columns as index so both axes show player names.


def _looks_like_range_index(idx):
    try:
        ints = [int(str(x)) for x in idx]
        return ints == list(range(len(idx)))
    except Exception:
        return False


def _is_numeric_string(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


if (
    _looks_like_range_index(matrix_pd.index)
    and matrix_pd.shape[0] == matrix_pd.shape[1]
    and any(not _is_numeric_string(c) for c in matrix_pd.columns)
):
    matrix_pd.index = matrix_pd.columns.copy()

# Create figure with exact dimensions used in project
fig, ax = plt.subplots(figsize=(FIG_WIDTH, HEATMAP_HEIGHT))

sns.heatmap(
    matrix_pd,
    ax=ax,
    cmap=HEATMAP_CMAP,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 2.2},
    square=True,
    linewidths=0.01,
    linecolor="#CCCCCC",
    cbar_kws={"label": "Jensen-Shannon Divergence", "shrink": 0.5, "pad": 0.04},
)

ax.set_xlabel("")
ax.set_ylabel("")

# Ensure tick labels use the player names
ax.set_xticklabels(
    [str(x).replace('"', "") for x in matrix_pd.columns],
    rotation=90,
    ha="center",
    fontsize=3.5,
)
ax.set_yticklabels(
    [str(y).replace('"', "") for y in matrix_pd.index], rotation=0, fontsize=3.5
)

plt.tight_layout()
fig.savefig(str(OUT), format="pdf", dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Saved heatmap to {OUT}")
