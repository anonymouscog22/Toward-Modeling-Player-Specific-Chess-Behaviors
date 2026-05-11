"""Training and inference utilities for UMAP-based dimensionality reduction.

This module trains a UMAP instance on latent vectors produced by the autoencoder
and applies the trained transformation to held-out data. Results (2D coordinates
with associated player identifiers) and the serialized UMAP model are persisted
to locations specified in the application configuration.

"""

import numpy as np
import polars as pl

from src.core.config import Config
from src.core.utils import getLogger
from src.features.umap import StyleUMAP

logger = getLogger()


def train_umap(config: Config) -> None:
    """Train a UMAP model on latent representations derived from the training set.

    The function loads precomputed latent vectors for the training split, fits a
    UMAP instance to these vectors to obtain a two-dimensional embedding, and
    persists both the embedding coordinates (as a Parquet file joined with the
    corresponding player identifiers) and the serialized UMAP model to disk.

    Parameters
    ----------
    config : Config
        Application configuration providing filesystem paths and UMAP hyperparameters.
    """
    vectors_path = config.paths.train_encoded_vectors_path
    result_path = config.paths.train_umap_result_path
    model_path = config.paths.umap_model_path

    logger.info(f"Loading training latent vectors from {vectors_path}")
    vectors = np.load(vectors_path, mmap_mode="r")

    logger.info("Fitting UMAP for dimensionality reduction...")
    model_umap = StyleUMAP(
        n_components=config.umap.n_components,
        n_jobs=-1,
        low_memory=False,
        verbose=True,
    )
    result_umap = model_umap.fit_transform(vectors)

    df = pl.read_parquet(config.paths.train_set_path)
    result_df = pl.DataFrame(result_umap, schema=["UMAP1", "UMAP2"])
    result_df = result_df.with_columns(df["player_name"])

    result_df.write_parquet(result_path)
    logger.info(f"Training UMAP results saved to {result_path}")

    model_umap.save_model(model_path)
    logger.info(f"UMAP model serialized to {model_path}")


def infer_umap(config: Config) -> None:
    """Apply a pre-trained UMAP model to the test (held-out) latent vectors.

    This function loads the serialized UMAP model, transforms the test latent
    vectors to obtain 2D coordinates, and saves the coordinates together with
    player identifiers to a Parquet file for downstream evaluation or plotting.
    """
    vectors_path = config.paths.test_encoded_vectors_path
    result_path = config.paths.test_umap_result_path
    model_path = config.paths.umap_model_path

    logger.info(f"Loading test latent vectors from {vectors_path}")
    vectors = np.load(vectors_path, mmap_mode="r")

    logger.info("Loading serialized UMAP model...")
    model_umap = StyleUMAP.load_model(model_path)

    logger.info("Applying UMAP transformation to unseen data...")
    result_umap = model_umap.transform(vectors)

    df = pl.read_parquet(config.paths.test_set_path)
    result_df = pl.DataFrame(result_umap, schema=["UMAP1", "UMAP2"])
    result_df = result_df.with_columns(df["player_name"])

    result_df.write_parquet(result_path)
    logger.info(f"Test UMAP results saved to {result_path}")


def run_umap_pipeline(config: Config) -> None:
    """Execute the complete UMAP pipeline: training followed by test-time inference.

    This convenience function fits a UMAP model on the training latent vectors
    and subsequently applies the fitted model to the test latent vectors,
    persisting both embedding results and the serialized model.
    """
    train_umap(config)
    infer_umap(config)
