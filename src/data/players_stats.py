"""Module for extracting per-player statistics from PGN archives.

This module iterates over the configured players' PGN files, computes summary
statistics (e.g. number of games, number of plies, mean year of play) and
persists the aggregated results as a Parquet file. All logging messages are
expressed in formal academic English to ensure clarity and reproducibility of
instrumentation traces.
"""

from pathlib import Path

import chess.pgn as pgn
import polars as pl
import tqdm

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def extract_players_stats(config: Config) -> None:
    """Extract and persist per-player summary statistics.

    This function traverses the raw PGN directories for each player declared in
    the provided configuration, computes basic aggregate statistics for each
    player's corpus (number of games, approximate number of plies, and the
    mean year of games when available), constructs a Polars DataFrame and
    persists the result to the configured player statistics path.

    Args:
        config: The validated pipeline configuration object containing file
            system paths and player identifiers.

    Returns:
        None. The resulting DataFrame is written to disk as a Parquet file.
    """
    data = []
    raw_data_dir = Path(config.paths.raw_data)

    for player_id, player_name in config.data.players.items():
        logger.info(
            "Extracting statistical summaries for player %s (ID: %s).",
            player_name,
            player_id,
        )

        pgn_folder = raw_data_dir / player_id

        if not pgn_folder.exists():
            logger.warning(
                "No PGN directory was located for player %s (ID: %s); this player will be skipped.",
                player_name,
                player_id,
            )
            continue

        pgn_files = list(pgn_folder.glob("*.pgn"))
        progress_bar = tqdm.tqdm(pgn_files, desc=f"Processing games for {player_name}")

        n_games = len(pgn_files)
        n_plys = 0
        mean_year = 0
        n_year = 0

        for pgn_path in progress_bar:
            with open(pgn_path, encoding="utf-8") as f:
                game = pgn.read_game(f)

            if game is None:
                logger.warning(
                    "Encountered an unreadable PGN file: %s for player %s; skipping.",
                    pgn_path.name,
                    player_name,
                )
                continue

            # PlyCount header stores half-move counts; convert to full moves
            n_plys += int(int(game.headers.get("PlyCount", 0)) / 2)

            year = (
                int(game.headers.get("Date", "????.??.??")[:4])
                if "Date" in game.headers
                else 0
            )
            if year != 0:
                mean_year += year
                n_year += 1

        mean_year /= n_year if n_year > 0 else 1

        data.append(
            {
                "player_id": player_id,
                "player_name": player_name,
                "n_games": n_games,
                "n_plys": n_plys,
                "mean_year": int(mean_year),
            }
        )

    logger.info("Constructing a Polars DataFrame from the aggregated statistics.")
    df = pl.DataFrame(data)
    logger.info("Player statistics DataFrame:\n%s", df.to_string())

    logger.info("Persisting the computed player statistics to disk.")
    df.write_parquet(config.paths.player_stats_path)

    logger.info(
        "Player statistics have been successfully written to %s.",
        config.paths.player_stats_path,
    )
