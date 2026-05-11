"""Module for extracting opening statistics (ECO codes) from PGN files.

This module parses PGN files for configured players, extracts the ECO code
for each game and records the player's color when the game was played. The
aggregated statistics are written to a Parquet file.
"""

from pathlib import Path

import chess.pgn as pgn
import polars as pl
import tqdm

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def extract_opening_stats(config: Config) -> None:
    """Extract opening statistics (ECO codes) from PGN files and persist them.

    For each configured player, parse PGN files, determine the player's color,
    extract the ECO code for the game and aggregate the results into a Polars
    DataFrame which is then written to Parquet according to configuration.
    """
    data = []
    raw_data_dir = Path(config.paths.raw_data)

    for player_id, player_name in config.data.players.items():
        logger.info(
            f"Extracting opening statistics for {player_name} (ID: {player_id})"
        )

        pgn_folder = raw_data_dir / player_id

        if not pgn_folder.exists():
            logger.warning(
                f"No PGN directory found for {player_name} (ID: {player_id}). Skipping."
            )
            continue

        pgn_files = list(pgn_folder.glob("*.pgn"))
        progress_bar = tqdm.tqdm(pgn_files, desc=f"Processing {player_name}")

        for pgn_path in progress_bar:
            with open(pgn_path, encoding="utf-8") as f:
                game = pgn.read_game(f)

            if game is None:
                continue

            white_player = game.headers.get("White", "")
            black_player = game.headers.get("Black", "")

            player_color = "Unknown"
            if player_name in white_player:
                player_color = "White"
            elif player_name in black_player:
                player_color = "Black"

            opening = game.headers.get("ECO", "Unknown")

            data.append(
                {
                    "player_name": player_name,
                    "player_color": player_color,
                    "opening": opening,
                }
            )

    logger.info("Generating the Polars DataFrame...")
    df = pl.DataFrame(data)

    logger.info("Saving opening statistics...")
    df.write_parquet(config.paths.opening_stats_path)

    logger.info("Opening statistics saved successfully.")
