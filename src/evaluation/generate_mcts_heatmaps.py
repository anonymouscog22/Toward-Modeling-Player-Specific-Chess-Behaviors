"""Generate AE->UMAP->JSD heatmaps for each predictions file produced by the MCTS grid search.

This utility discovers parquet files produced by `evaluate_mcts_params` in the
configured `paths.evaluation_dir` (by default files named like
`mcts_grid_sim{n}_c{c}_thr{t}.parquet`) and for each:

- Builds AE->UMAP 2D embeddings treating the MCTS predictions as a model variant
- Computes test-pairwise distances (JSD), train-vs-test stability and full cross-matrix
- Emits per-method JSD heatmap and stability heatmap using the existing
  visualization utilities (templates in `config.paths` are used)

Usage
-----
Run from the project root:

    python -m src.evaluation.generate_mcts_heatmaps --config config/default.yml

"""

import argparse
import logging
import re
from pathlib import Path

import polars as pl

from src.core.config import Config
from src.core.utils import getLogger
from src.evaluation.compute_distances import (
    compute_distances,
    compute_full_cross_matrix,
    compute_train_test_distances,
)
from src.evaluation.compute_model_jsd import build_and_save_model_umap_for_variant
from src.visualization.graphics import generate_model_graphics

logger = getLogger() or logging.getLogger(__name__)

# Allow underscores inside numeric components (e.g. c0_5 representing 0.5)
MCTS_PATTERN = re.compile(
    r"mcts_grid_sim(?P<sim>[^_]+)_c(?P<c>[^_]+(?:_[^_]+)*)_thr(?P<thr>[^.]+)\.parquet$"
)


def discover_mcts_files(eval_dir: Path, prefix: str = "mcts_grid") -> list:
    """Return list of parquet files in `eval_dir` matching the MCTS grid prefix.

    Returns the list of Path objects (unsorted) and logs the candidates for
    easier debugging when the environment/CWD changes (e.g. under `uv run`).
    """
    pattern = f"{prefix}_sim*_c*_thr*.parquet"
    candidates = sorted(eval_dir.glob(pattern)) if eval_dir.exists() else []

    # Log candidate filenames for debugging
    try:
        logger.info(
            "discover_mcts_files: eval_dir=%s, pattern=%s, candidates_count=%d",
            eval_dir,
            pattern,
            len(candidates),
        )
        logger.info("discover_mcts_files: candidates=%s", [p.name for p in candidates])
    except Exception:
        pass

    return [p for p in candidates if MCTS_PATTERN.search(p.name)]


def sanitize_component(s: str) -> str:
    """Make a string safe for use in method identifiers (replace dots with underscore)."""
    return s.replace(".", "_")


def process_file(config: Config, path: Path, device: str = "cuda") -> None:
    """Process a single MCTS predictions parquet and generate heatmaps."""
    m = MCTS_PATTERN.search(path.name)
    if not m:
        logger.warning("Skipping file with unexpected name: %s", path)
        return

    sim = sanitize_component(m.group("sim"))
    c = sanitize_component(m.group("c"))
    thr = sanitize_component(m.group("thr"))

    out_method = f"maia2_ft_mcts_sim{sim}_c{c}_thr{thr}"
    logger.info("Processing %s -> method id: %s", path.name, out_method)

    preds_df = pl.read_parquet(str(path))

    # Build embeddings (train split for the synthetic method)
    build_and_save_model_umap_for_variant(
        config, preds_df, pred_col="pred_mcts", out_method=out_method, device=device
    )

    # Compute distances (test pairwise), cross-matrix and train/test stability
    kde = config.jsd.kde
    compute_distances(config, method=out_method, is_test=True, kde=kde)
    compute_train_test_distances(config, method=out_method, kde=kde)
    compute_full_cross_matrix(config, method=out_method, kde=kde)

    # Generate figures for this method using the graphics helper
    try:
        generate_model_graphics(config, methods=[out_method])
    except Exception as exc:  # pragma: no cover - continue processing others
        logger.error("Failed to generate graphics for %s: %s", out_method, exc)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate heatmaps for MCTS grid results"
    )
    parser.add_argument(
        "--config", type=str, default="config/default.yml", help="Path to YAML config"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="mcts_grid",
        help="Filename prefix used by evaluate_mcts_params",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the autoencoder on (cpu|cuda). If omitted, auto-detects",
    )

    args = parser.parse_args(argv)

    config = Config.from_yaml(args.config)
    eval_dir = Path(config.paths.evaluation_dir)

    device = args.device
    if device is None:
        # Simple detection: use CUDA if available via torch
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    files = discover_mcts_files(eval_dir, prefix=args.prefix)

    # Helpful debug info: log configured evaluation dir and current working directory
    from os import getcwd

    logger.info("Configured evaluation_dir: %s", config.paths.evaluation_dir)
    logger.info("Current working directory: %s", getcwd())

    # Fallback: if no files found in the configured eval dir, search the repository tree for matching files.
    if not files:
        repo_root = Path(__file__).resolve().parents[2]
        logger.warning(
            "No MCTS grid result files found in %s with prefix %s. Trying repository-wide search from %s",
            eval_dir,
            args.prefix,
            repo_root,
        )
        files = sorted(repo_root.rglob(f"{args.prefix}_sim*_c*_thr*.parquet"))
        # Keep only those matching the strict pattern
        files = [p for p in files if MCTS_PATTERN.search(p.name)]

    if not files:
        logger.warning(
            "Still no MCTS grid result files found. Please verify that evaluate_mcts_params has run and produced files named like '%s_sim...'.",
            args.prefix,
        )
        return

    logger.info("Found %d MCTS result files. Generating heatmaps...", len(files))

    for p in files:
        try:
            process_file(config, p, device=device)
        except Exception as exc:  # pragma: no cover - continue on errors
            logger.error("Error processing %s: %s", p, exc)

    logger.info("Completed generating heatmaps for MCTS grid.")


if __name__ == "__main__":
    main()
