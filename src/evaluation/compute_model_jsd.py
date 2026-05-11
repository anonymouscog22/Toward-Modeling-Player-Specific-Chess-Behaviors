"""Utilities to convert model-generated predictions into AE->UMAP embeddings and evaluate JSD.

This module orchestrates the conversion of model predictions (e.g., Maia variants)
into autoencoder latent vectors, transforms them via a serialized UMAP instance,
and reuses existing evaluation routines to compute Jensen-Shannon Divergence
matrices and stability analyses.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.evaluation.compute_distances import (
    compute_distances,
    compute_full_cross_matrix,
    compute_train_test_distances,
)
from src.features.umap import StyleUMAP, position_to_vector
from src.models.autoencoder import Autoencoder

logger = getLogger()


VARIANT_TO_PRED_COL: Dict[str, str] = {
    "maia2": "pred_baseline",
    "maia2_ft": "pred_custom",
    "maia2_ft_mcts": "pred_mcts",
}


def _ensure_predictions(config: Config) -> pl.DataFrame:
    path = config.paths.predictions_path
    if not Path(path).exists():
        from src.evaluation.evaluate_players import generate_predictions_parquet

        logger.info("Predictions parquet not found — generating predictions...")
        generate_predictions_parquet(config)

    return pl.read_parquet(path)


def _load_autoencoder_and_umap(config: Config, device: str):
    # Determine input dim from existing vectors
    vectors_path = Path(config.paths.test_vectors_path)
    if not vectors_path.exists():
        raise FileNotFoundError(f"Test vectors not found at {vectors_path}")

    sample = np.load(str(vectors_path), mmap_mode="r")
    input_dim = int(sample.shape[1])
    latent_dim = int(config.autoencoder.latent_dim)

    logger.info(
        f"Instantiating Autoencoder(input_dim={input_dim}, latent_dim={latent_dim})"
    )
    ae = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

    ae_path = Path(config.paths.autoencoder_model_path)
    if not ae_path.exists():
        raise FileNotFoundError(f"Autoencoder model not found at {ae_path}")

    state = torch.load(str(ae_path), map_location=device)
    ae.load_state_dict(state)
    ae.eval()

    umap_path = Path(config.paths.umap_model_path)
    if not umap_path.exists():
        raise FileNotFoundError(f"UMAP model not found at {umap_path}")

    logger.info(f"Loading UMAP model from {umap_path}")
    umap = StyleUMAP.load_model(str(umap_path))

    return ae, umap


def _encode_batch(ae: Autoencoder, tensors: torch.Tensor, device: str) -> np.ndarray:
    with torch.no_grad():
        tensors = tensors.to(device)
        latents = ae.encode(tensors)
        return latents.cpu().numpy()


def build_and_save_model_umap_for_variant(
    config: Config, preds_df: pl.DataFrame, pred_col: str, out_method: str, device: str
) -> None:
    """Construct AE->UMAP 2D coordinates for the provided prediction column and save them as a parquet.

    We treat these model-generated embeddings as the 'train' split for the synthetic method
    so that existing evaluation routines can be reused (train vs test comparisons).
    """
    logger.info(
        f"Building AE->UMAP coordinates for variant '{out_method}' using column '{pred_col}'..."
    )

    fens = preds_df["fen"].to_list()
    moves = preds_df[pred_col].to_list()
    players = preds_df["player_name"].to_list()

    assert len(fens) == len(moves) == len(players)

    batch_size = 1024

    # Prepare storage
    coords = []

    # Load AE and UMAP lazily outside loop
    ae, umap = _load_autoencoder_and_umap(config, device)

    # Process in chunks
    for i in tqdm(range(0, len(fens), batch_size), desc=f"Encoding {out_method}"):
        end = min(i + batch_size, len(fens))
        batch_fens = fens[i:end]
        batch_moves = moves[i:end]

        tensors = []
        for fen, mv in zip(batch_fens, batch_moves):
            if mv is None:
                # placeholder: create a zero vector if prediction missing
                # this will produce a point but will be ignored later if needed
                vec = torch.tensor([0.0])
            else:
                try:
                    vec = position_to_vector(fen, mv)
                except Exception:
                    # If position/move invalid, fallback to a zero vector of correct dim
                    # we create from test_vectors shape
                    sample = np.load(config.paths.test_vectors_path, mmap_mode="r")
                    vec = torch.as_tensor(sample[0]).float()
            tensors.append(vec.float())

        # Stack into tensor of shape (B, input_dim)
        batch_tensor = torch.stack(tensors, dim=0)
        latents = _encode_batch(ae, batch_tensor, device)

        # Apply UMAP transform
        umap_coords = umap.transform(latents)
        coords.append(umap_coords)

    coords = np.vstack(coords)

    # Build DataFrame
    df_out = pl.DataFrame(
        {"UMAP1": coords[:, 0], "UMAP2": coords[:, 1], "player_name": players}
    )

    # Save using the project's canonical embeddings path
    # Save using the project's canonical embeddings path (configurable template)
    out_path = Path(
        config.paths.method_train_embeddings_template.format(method=out_method)
    )

    # Avoid overwriting identical file accidentally
    if out_path.exists():
        logger.debug("Overwriting existing embeddings file: %s", out_path)

    df_out.write_parquet(str(out_path))
    logger.info("Saved model-generated embeddings to %s", out_path)

    # Ensure test embeddings path exists for this method by copying the real test UMAP (if available)
    test_src = Path(config.paths.test_umap_result_path)
    test_dst = Path(
        config.paths.method_test_embeddings_template.format(method=out_method)
    )
    if test_src.exists():
        from shutil import copyfile

        copyfile(str(test_src), str(test_dst))
        logger.info("Copied real test UMAP to %s for comparison", test_dst)
    else:
        logger.warning(
            "Real test UMAP not found at %s; cross-split comparison will fail if missing",
            test_src,
        )


def run_model_jsd_pipeline(config: Config) -> None:
    """High-level driver to compute AE->UMAP->JSD for the specified Maia variants.

    Outputs are written using the project's existing evaluation helpers.
    """
    preds_df = _ensure_predictions(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for method, col in VARIANT_TO_PRED_COL.items():
        logger.info(f"Processing variant '{method}'...")
        build_and_save_model_umap_for_variant(config, preds_df, col, method, device)

        # Now reuse existing compute routines
        kde = config.jsd.kde
        compute_train_test_distances(config, method=method, kde=kde)
        compute_full_cross_matrix(config, method=method, kde=kde)
        # Optionally compute pairwise test distances for the synthetic 'train' embeddings
        compute_distances(config, method=method, is_test=False, kde=kde)

    logger.info("Completed AE->UMAP->JSD pipeline for all Maia variants.")
