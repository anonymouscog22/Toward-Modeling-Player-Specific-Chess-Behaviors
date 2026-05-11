"""Compute vector representations for chess positions.

This module provides utilities to extract fixed-size vector representations for
(FEN, move) pairs stored in Parquet datasets. Each pair is converted via
`position_to_vector`. The module writes NumPy arrays for train/test sets.
"""

from pathlib import Path

import numpy as np
import polars as pl
import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.features.umap import position_to_vector

logger = getLogger()


def _extract_and_save(input_path: Path, output_path: Path, desc: str) -> None:
    """Utility helper to extract vectors from a specific Parquet file and persist them.

    Loads a Parquet dataset from `input_path`, converts each (FEN, move) pair to a
    vector using `position_to_vector`, and writes the resulting contiguous NumPy
    array to `output_path`.

    Parameters
    ----------
    input_path : Path
        Path to the input Parquet dataset.
    output_path : Path
        Destination path for the NumPy array (.npy).
    desc : str
        Short descriptor used in progress messages (e.g. 'TRAIN' or 'TEST').
    """
    if not input_path.exists():
        logger.error(f"Dataset file not found: {input_path}")
        return

    df = pl.read_parquet(input_path)
    n_rows = len(df)

    logger.info(f"[{desc}] Computing vectors for {n_rows} positions...")

    sample_vec = position_to_vector(df["fen"][0], df["move"][0])
    vector_dim = sample_vec.shape[0]

    # Pre-allocate memory for the resulting vectors
    all_vectors = np.zeros((n_rows, vector_dim), dtype=np.float32)

    for i, row in enumerate(
        tqdm.tqdm(
            df.iter_rows(named=True),
            total=n_rows,
            desc=f"Extracting vectors ({desc})",
            unit="vector",
        )
    ):
        try:
            vec = position_to_vector(row["fen"], row["move"])
            all_vectors[i] = vec.cpu().numpy().astype(np.float32)
        except Exception as e:
            logger.warning(f"Error processing row {i}: {e}")

    np.save(output_path, all_vectors)
    logger.info(f"[{desc}] Vectors saved to {output_path}")


def compute_vectors(config: Config) -> None:
    """Compute vectors separately for the training and test datasets."""
    # 1. Process the training set
    _extract_and_save(
        Path(config.paths.train_set_path),
        Path(config.paths.train_vectors_path),
        "TRAIN",
    )

    # 2. Process the test set
    _extract_and_save(
        Path(config.paths.test_set_path), Path(config.paths.test_vectors_path), "TEST"
    )

    logger.info("Extraction of all vectors completed successfully.")
