"""Evaluation utilities for computing stylistic distances between player embeddings.

This module provides routines to compute pairwise Jensen-Shannon divergences
between empirical 2D embedding distributions and to assess the stability of
embedding methods by comparing training and test splits.
"""

from itertools import combinations

import numpy as np
import polars as pl
import tqdm
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def compute_js_distance_continuous(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute the Jensen-Shannon divergence between two empirical ND distributions using KDE.
    """
    kde_p = gaussian_kde(emb1.T)
    kde_q = gaussian_kde(emb2.T)

    eps = 1e-10

    p_eval_p = kde_p(emb1.T)
    q_eval_p = kde_q(emb1.T)
    m_eval_p = 0.5 * (p_eval_p + q_eval_p)

    kl_pm = np.mean(np.log2(p_eval_p + eps) - np.log2(m_eval_p + eps))

    p_eval_q = kde_p(emb2.T)
    q_eval_q = kde_q(emb2.T)
    m_eval_q = 0.5 * (p_eval_q + q_eval_q)

    kl_qm = np.mean(np.log2(q_eval_q + eps) - np.log2(m_eval_q + eps))

    js_divergence = 0.5 * kl_pm + 0.5 * kl_qm
    js_divergence = max(0.0, js_divergence)
    return np.sqrt(js_divergence)


def compute_js_distance(
    emb1: np.ndarray, emb2: np.ndarray, bins: int = 15, bounds: list | None = None
) -> float:
    """Compute the Jensen-Shannon divergence between two empirical 2D distributions.

    The function constructs 2D histograms for each embedding (using `bins` and
    optional `bounds`), flattens the histograms into discrete probability
    vectors and computes the Jensen-Shannon divergence with base-2 logarithm.
    """
    hist1, _, _ = np.histogram2d(emb1[:, 0], emb1[:, 1], bins=bins, range=bounds)
    hist2, _, _ = np.histogram2d(emb2[:, 0], emb2[:, 1], bins=bins, range=bounds)

    p = hist1.flatten()
    q = hist2.flatten()

    return jensenshannon(p, q, base=2)


def _get_dim_columns(df: pl.DataFrame) -> list:
    """Identify numeric columns corresponding to latent embedding dimensions."""
    return [col for col in df.columns if col not in ["player_name", "game_id"]]


def compute_distances(
    config: Config, method: str, is_test: bool = False, kde: bool = True
) -> None:
    """Compute stylistic distances between players for a specified embedding method."""

    input_path = config.paths.get_embeddings_path(method, is_test)
    output_path = config.paths.get_distances_path(method, is_test, kde)

    logger.info("Loading %s representations from %s...", method.upper(), input_path)
    df = pl.read_parquet(input_path)

    cols = _get_dim_columns(df)
    distance_data = []

    player_names = df["player_name"].unique().to_list()

    # Determine global boundaries to ensure a consistent grid for 2D histograms
    global_bounds = [
        [df[cols[0]].min(), df[cols[0]].max()],
        [df[cols[1]].min(), df[cols[1]].max()],
    ]

    progress_bar = tqdm.tqdm(
        list(combinations(player_names, 2)), desc=f"JS comparison ({method.upper()})"
    )

    for p1, p2 in progress_bar:
        emb1 = df.filter(pl.col("player_name") == p1).select(cols).to_numpy()
        emb2 = df.filter(pl.col("player_name") == p2).select(cols).to_numpy()

        if len(emb1) == 0 or len(emb2) == 0:
            continue

        if kde:
            distance_js = compute_js_distance_continuous(emb1, emb2)
        else:
            distance_js = compute_js_distance(emb1, emb2, bounds=global_bounds)
        distance_data.append({"p1": p1, "p2": p2, "distance": distance_js})

    distance_df = pl.DataFrame(distance_data)
    logger.info("Saving computed distances to %s", output_path)
    distance_df.write_parquet(output_path)


def compute_train_test_distances(
    config: Config,
    method: str,
    kde: bool = True,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
) -> None:
    """Evaluate the stability of an embedding method by comparing training and test splits.

    Optionally perform a bootstrap over embeddings for each player to produce
    confidence intervals (percentile CI) for the train/test Jensen-Shannon distance
    on the diagonal (one value per player).

    Parameters
    - n_bootstrap: number of bootstrap resamples to perform per player (0 -> no bootstrap)
    - ci_alpha: two-sided alpha for percentile CI (default 0.05 -> 95% CI)
    """

    train_path = config.paths.get_embeddings_path(method, is_test=False)
    test_path = config.paths.get_embeddings_path(method, is_test=True)
    output_path = config.paths.get_cross_distances_path(method, kde)

    logger.info("Loading training and test representations for %s...", method.upper())
    df_train = pl.read_parquet(train_path)
    df_test = pl.read_parquet(test_path)

    cols = _get_dim_columns(df_train)
    distance_data = []
    player_names = df_train["player_name"].unique().to_list()

    # Global bounds adjusted across both sets to ensure identical histogram grids
    global_bounds = [
        [
            min(df_train[cols[0]].min(), df_test[cols[0]].min()),
            max(df_train[cols[0]].max(), df_test[cols[0]].max()),
        ],
        [
            min(df_train[cols[1]].min(), df_test[cols[1]].min()),
            max(df_train[cols[1]].max(), df_test[cols[1]].max()),
        ],
    ]

    progress_bar = tqdm.tqdm(
        player_names, desc=f"Train/Test stability ({method.upper()})"
    )

    for player in progress_bar:
        emb_train = (
            df_train.filter(pl.col("player_name") == player).select(cols).to_numpy()
        )
        emb_test = (
            df_test.filter(pl.col("player_name") == player).select(cols).to_numpy()
        )

        if len(emb_train) == 0 or len(emb_test) == 0:
            continue

        # compute the point estimate on the full data
        try:
            if kde:
                distance_js = compute_js_distance_continuous(emb_train, emb_test)
            else:
                distance_js = compute_js_distance(
                    emb_train, emb_test, bounds=global_bounds
                )
        except Exception:
            logger.debug(
                "Failed to compute JSD for player %s (method=%s); marking NaN",
                player,
                method,
            )
            distance_js = float("nan")

        record = {"player": player, "distance": distance_js}

        # Optional bootstrap to compute CI for the diagonal distance
        if n_bootstrap and len(emb_train) > 0 and len(emb_test) > 0:
            bs_values = []
            for _ in range(n_bootstrap):
                # resample with replacement within each split, keeping same sample size
                try:
                    idx_t = np.random.choice(
                        len(emb_train), size=int(len(emb_train) * 0.8), replace=False
                    )
                    idx_e = np.random.choice(
                        len(emb_test), size=int(len(emb_test) * 0.8), replace=False
                    )
                    emb_t_bs = emb_train[idx_t]
                    emb_e_bs = emb_test[idx_e]

                    if kde:
                        # KDE may fail on extremely small / degenerate samples; wrap defensively
                        try:
                            d_bs = compute_js_distance_continuous(emb_t_bs, emb_e_bs)
                        except Exception:
                            # fallback to histogram-based distance using global bounds
                            d_bs = compute_js_distance(
                                emb_t_bs, emb_e_bs, bounds=global_bounds
                            )
                    else:
                        d_bs = compute_js_distance(
                            emb_t_bs, emb_e_bs, bounds=global_bounds
                        )

                    bs_values.append(float(d_bs))
                except Exception:
                    # skip failed bootstrap iteration
                    continue

            if bs_values:
                bs_arr = np.array(bs_values)
                bs_mean = float(np.mean(bs_arr))
                bs_std = (
                    float(np.std(bs_arr, ddof=1)) if bs_arr.size > 1 else float("nan")
                )
                lower = float(np.percentile(bs_arr, 100 * (ci_alpha / 2)))
                upper = float(np.percentile(bs_arr, 100 * (1 - ci_alpha / 2)))

                record.update(
                    {
                        "bs_mean": bs_mean,
                        "bs_std": bs_std,
                        f"ci_lower_{int((1 - ci_alpha) * 100)}": lower,
                        f"ci_upper_{int((1 - ci_alpha) * 100)}": upper,
                        "bs_n": int(bs_arr.size),
                    }
                )
            else:
                # no successful bootstrap draws
                record.update(
                    {
                        "bs_mean": float("nan"),
                        "bs_std": float("nan"),
                        f"ci_lower_{int((1 - ci_alpha) * 100)}": float("nan"),
                        f"ci_upper_{int((1 - ci_alpha) * 100)}": float("nan"),
                        "bs_n": 0,
                    }
                )

        distance_data.append(record)

    distance_df = pl.DataFrame(distance_data)
    logger.info("Saving cross-split analysis to %s", output_path)
    distance_df.write_parquet(output_path)


def compute_full_cross_matrix(config: Config, method: str, kde: bool = True) -> None:
    """Compute the full asymmetric cross-distance matrix between train and test players.

    This function computes distances for every ordered pair (train_player, test_player)
    using the specified embedding method and returns a table suitable for heatmap
    generation and further analysis.
    """
    train_path = config.paths.get_embeddings_path(method, is_test=False)
    test_path = config.paths.get_embeddings_path(method, is_test=True)

    output_path = config.paths.get_full_cross_matrix_path(method, kde)

    logger.info("Loading representations for full cross-matrix (%s)...", method.upper())
    df_train = pl.read_parquet(train_path)
    df_test = pl.read_parquet(test_path)

    cols = _get_dim_columns(df_train)
    distance_data = []

    players = df_train["player_name"].unique().to_list()

    # Determine global boundaries across both train and test sets to ensure identical histogram grids
    global_bounds = [
        [
            min(df_train[cols[0]].min(), df_test[cols[0]].min()),
            max(df_train[cols[0]].max(), df_test[cols[0]].max()),
        ],
        [
            min(df_train[cols[1]].min(), df_test[cols[1]].min()),
            max(df_train[cols[1]].max(), df_test[cols[1]].max()),
        ],
    ]

    # Nested iteration to construct the full asymmetric cross-distance matrix
    progress_bar = tqdm.tqdm(players, desc=f"Full Cross-Matrix ({method.upper()})")

    for p_train in progress_bar:
        emb_train = (
            df_train.filter(pl.col("player_name") == p_train).select(cols).to_numpy()
        )

        for p_test in players:
            emb_test = (
                df_test.filter(pl.col("player_name") == p_test).select(cols).to_numpy()
            )

            if len(emb_train) == 0 or len(emb_test) == 0:
                continue

            if kde:
                distance_js = compute_js_distance_continuous(emb_train, emb_test)
            else:
                distance_js = compute_js_distance(
                    emb_train, emb_test, bounds=global_bounds
                )

            distance_data.append(
                {"p_train": p_train, "p_test": p_test, "distance": distance_js}
            )

    distance_df = pl.DataFrame(distance_data)
    logger.info("Saving full cross matrix to %s", output_path)
    distance_df.write_parquet(output_path)
