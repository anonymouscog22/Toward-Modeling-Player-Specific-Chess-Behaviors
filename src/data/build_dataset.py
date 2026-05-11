"""Module for constructing a dataset from downloaded PGN files.

This module parses Portable Game Notation (PGN) files for players specified in the
project configuration, extracts per-move examples (position FEN, UCI move, player
color, move number, repetition flag and final result), and persists the full
dataset and a train/test split to Parquet files.
"""

from pathlib import Path

import chess
import polars as pl
import tqdm
from chess import pgn
from sklearn.model_selection import train_test_split

from src.core.config import Config
from src.core.utils import getLogger

logger = getLogger()


def build_dataset(config: Config) -> None:
    """Construct a dataset from downloaded PGN files for each configured player.

    The function iterates through PGN files stored under the configured raw data
    directory, extracts board positions and corresponding moves for positions where
    the configured player is to move, and records associated metadata. The full
    dataset is saved to a Parquet file and a randomized train/test split is written
    according to the configuration.

    Parameters
    ----------
    config : Config
        Application configuration providing player definitions, filesystem paths,
        HTTP and concurrency settings.
    """
    data = []
    data_count = {}

    raw_data_dir = Path(config.paths.raw_data)

    for player_id, player_name in config.data.players.items():
        logger.info(f"Converting PGN files for {player_name} (ID: {player_id})")

        pgn_folder = raw_data_dir / player_id

        if not pgn_folder.exists():
            logger.warning(
                f"No PGN directory found for {player_name} (ID: {player_id}). Skipping."
            )
            continue

        pgn_files = list(pgn_folder.glob("*.pgn"))
        progress_bar = tqdm.tqdm(pgn_files, desc=f"Processing {player_name}")

        for pgn_path in progress_bar:
            with open(pgn_path, "r", encoding="utf-8") as f:
                game = pgn.read_game(f)

            if game is None:
                logger.warning(
                    f"Unable to read game {pgn_path.name} for {player_name}. Skipping."
                )
                continue

            header = game.headers
            white_player = header.get("White", "")
            black_player = header.get("Black", "")

            player_color = None

            if player_name in white_player:
                player_color = chess.WHITE
            elif player_name in black_player:
                player_color = chess.BLACK
            else:
                logger.warning(
                    f"Player {player_name} not found in headers of {pgn_path.name}. Skipping."
                )
                continue

            result = header.get("Result", "")
            if result not in ["1-0", "0-1", "1/2-1/2"]:
                logger.warning(
                    f"Unexpected result '{result}' in {pgn_path.name} for {player_name}. Skipping."
                )
                continue

            data_count[player_name] = data_count.get(player_name, 0) + 1

            # Exclude non-standard chess variants (e.g., Chess960 or puzzles)
            # which are indicated by the presence of a custom FEN header.
            fen = header.get("FEN", None)
            if fen is not None:
                logger.warning(
                    f"Game {pgn_path.name} for {player_name} already contains a FEN header. Skipping."
                )
                continue

            board = game.board()

            # Reconstruct the board state at each step to extract valid (FEN, move)
            # pairs strictly from the perspective of the player of interest.
            for move in game.mainline_moves():
                if board.turn == player_color:
                    color = "white" if board.turn == chess.WHITE else "black"
                    current_fen = board.fen()
                    move_uci = move.uci()

                    data.append(
                        {
                            "game_id": pgn_path.stem,
                            "round": board.fullmove_number,
                            "player_name": player_name,
                            "player_color": color,
                            "fen": current_fen,
                            "move": move_uci,
                            "repetition": board.is_repetition(2),
                            "result": result,
                        }
                    )
                board.push(move)

    logger.info("Generating Polars DataFrame...")

    df = pl.DataFrame(data, schema=config.data.dataset_col_order)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    assert isinstance(df_train, pl.DataFrame)
    assert isinstance(df_test, pl.DataFrame)

    logger.info("Saving Parquet files...")

    df_train.write_parquet(config.paths.train_set_path)
    df_test.write_parquet(config.paths.test_set_path)
    df.write_parquet(config.paths.dataset_path)

    logger.info("Dataset creation completed successfully.")
