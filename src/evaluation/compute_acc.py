"""Compute and persist per-player accuracy metrics.

This module reads the predictions parquet produced by the evaluation pipeline,
computes top-1 accuracy for various prediction methods per player, and writes
the aggregated accuracy table to disk for downstream reporting.
"""

import polars as pl

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def compute_accuracy(config: Config) -> None:
    """Compute and persist per-player accuracy statistics.

    Parameters
    ----------
    config : Config
        Project configuration providing the path to the predictions parquet and
        the target path for accuracy results.
    """
    df = pl.read_parquet(config.paths.predictions_path)

    df_results = []

    for player in df["player_name"].unique():
        player_df = df.filter(pl.col("player_name") == player)
        baseline_accuracy = (
            player_df["pred_baseline"] == player_df["true_move"]
        ).mean()
        custom_accuracy = (player_df["pred_custom"] == player_df["true_move"]).mean()
        mcts_accuracy = (player_df["pred_mcts"] == player_df["true_move"]).mean()

        df_results.append(
            {
                "player_name": player,
                "baseline_accuracy": baseline_accuracy,
                "custom_accuracy": custom_accuracy,
                "mcts_accuracy": mcts_accuracy,
            }
        )

    df_results = pl.DataFrame(df_results)
    df_results.write_parquet(config.paths.accuracy_path)
    logger.info(
        "Player accuracies have been computed and written to %s",
        config.paths.accuracy_path,
    )
