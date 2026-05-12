"""Compute and persist per-player accuracy metrics.

This module reads the predictions parquet produced by the evaluation pipeline,
computes top-1 accuracy for various prediction methods per player, and writes
the aggregated accuracy table to disk for downstream reporting.

Added: optional subsample-based bootstrap (similar to compute_distances.py)
which produces mean, std, and percentile confidence intervals for each method.
"""

import numpy as np
import polars as pl

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def compute_accuracy(
    config: Config, n_bootstrap: int = 1000, ci_alpha: float = 0.05
) -> None:
    """Compute and persist per-player accuracy statistics.

    Parameters
    ----------
    config : Config
        Project configuration providing the path to the predictions parquet and
        the target path for accuracy results.

    n_bootstrap : int
        Number of subsample bootstrap iterations to perform per player. If 0,
        no bootstrap statistics are computed.

    ci_alpha : float
        Two-sided alpha for percentile confidence intervals (default 0.05 -> 95%% CI).
    """
    df = pl.read_parquet(config.paths.predictions_path)

    df_results = []

    for player in df["player_name"].unique():
        player_df = df.filter(pl.col("player_name") == player)

        # point estimates (ensure floats)
        baseline_accuracy = float(
            (player_df["pred_baseline"] == player_df["true_move"]).mean()
        )
        custom_accuracy = float(
            (player_df["pred_custom"] == player_df["true_move"]).mean()
        )
        mcts_accuracy = float((player_df["pred_mcts"] == player_df["true_move"]).mean())

        record = {
            "player_name": player,
            "baseline_accuracy": baseline_accuracy,
            "custom_accuracy": custom_accuracy,
            "mcts_accuracy": mcts_accuracy,
        }

        # Optional subsample-based bootstrap to estimate variability and CI
        if n_bootstrap and len(player_df) > 0:
            # Convert to numpy for fast subsampling
            arr = player_df.select(
                ["pred_baseline", "pred_custom", "pred_mcts", "true_move"]
            ).to_numpy()

            bs_baseline = []
            bs_custom = []
            bs_mcts = []

            for _ in range(n_bootstrap):
                try:
                    # match compute_distances approach: subsample ~80%% without replacement
                    idx = np.random.choice(
                        len(arr), size=max(1, int(len(arr) * 0.8)), replace=False
                    )
                    sample = arr[idx]

                    true = sample[:, 3]

                    b_baseline = float(np.mean(sample[:, 0] == true))
                    b_custom = float(np.mean(sample[:, 1] == true))
                    b_mcts = float(np.mean(sample[:, 2] == true))

                    bs_baseline.append(b_baseline)
                    bs_custom.append(b_custom)
                    bs_mcts.append(b_mcts)
                except Exception:
                    # skip failed subsample
                    continue

            def summarize_bs(bs_values: list, name_prefix: str):
                if bs_values:
                    arr_bs = np.array(bs_values)
                    mean = float(np.mean(arr_bs))
                    std = (
                        float(np.std(arr_bs, ddof=1))
                        if arr_bs.size > 1
                        else float("nan")
                    )
                    lower = float(np.percentile(arr_bs, 100 * (ci_alpha / 2)))
                    upper = float(np.percentile(arr_bs, 100 * (1 - ci_alpha / 2)))
                    return {
                        f"{name_prefix}_bs_mean": mean,
                        f"{name_prefix}_bs_std": std,
                        f"{name_prefix}_ci_lower_{int((1 - ci_alpha) * 100)}": lower,
                        f"{name_prefix}_ci_upper_{int((1 - ci_alpha) * 100)}": upper,
                        f"{name_prefix}_bs_n": int(arr_bs.size),
                    }
                else:
                    return {
                        f"{name_prefix}_bs_mean": float("nan"),
                        f"{name_prefix}_bs_std": float("nan"),
                        f"{name_prefix}_ci_lower_{int((1 - ci_alpha) * 100)}": float(
                            "nan"
                        ),
                        f"{name_prefix}_ci_upper_{int((1 - ci_alpha) * 100)}": float(
                            "nan"
                        ),
                        f"{name_prefix}_bs_n": 0,
                    }

            record.update(summarize_bs(bs_baseline, "baseline"))
            record.update(summarize_bs(bs_custom, "custom"))
            record.update(summarize_bs(bs_mcts, "mcts"))

        df_results.append(record)

    df_results = pl.DataFrame(df_results)
    df_results.write_parquet(config.paths.accuracy_path)
    logger.info(
        "Player accuracies (with optional bootstrap stats) have been computed and written to %s",
        config.paths.accuracy_path,
    )
