# Toward Modeling Player-Specific Chess Behaviors

This repository contains the code and artifacts accompanying the paper "Toward Modeling Player-Specific Chess Behaviors." The primary objective of the project is to develop and evaluate models that reproduce individual human chess-playing styles and to introduce robust metrics for assessing stylistic fidelity.

## Abstract

While artificial intelligence has achieved superhuman performance in chess, developing models that accurately emulate the individualized decision-making styles of human players remains a significant challenge. Existing human-like chess models capture general population behaviors based on skill levels but fail to reproduce the behavioral characteristics of specific historical champions. Furthermore, the standard evaluation metric, move accuracy, inherently penalizes natural human variance and ignores long-term behavioral consistency, leading to an incomplete assessment of stylistic fidelity. To address these limitations, an architecture is proposed that adapts the unified Maia-2 model to champion-specific embeddings, further enhanced by the integration of a limited Monte Carlo Tree Search (MCTS) process to enrich tactical exploration during move selection. To robustly evaluate this approach, a novel behavioral metric based on the Jensen-Shannon divergence is introduced. By compressing high-dimensional board representations into a latent space using an AutoEncoder and Uniform Manifold Approximation and Projection (UMAP), move distributions are discretized on a common grid to compare behavioral similarities. Results across 16 historical world champions indicate that while integrating MCTS decreases standard deterministic move accuracy, it improves stylistic alignment according to the proposed metric, substantially reducing the average Jensen-Shannon divergence. Ultimately, the proposed metric successfully discriminates between individual players and provides promising evidence toward more comprehensive evaluations of human-AI behavioral alignment.

## Contents

This README provides:
- Software prerequisites and installation instructions.
- A quickstart to reproduce typical workflows (data acquisition → training → evaluation → visualization).
- A description of the configuration system and how to customize it.
- A detailed explanation of the pipeline stages and the commands that implement them.
- Notes about outputs, reproducibility, and contribution guidelines.

## Requirements

- Python 3.13 is the supported runtime. Use a virtual environment to isolate dependencies.

- Optionally, the project can be managed with `uv` (workspace manager). When using `uv`, install and synchronize dependencies with:

```bash
uv sync
```

If dependency resolution fails with an error such as "No solution found when resolving dependencies ...", re-run with a permissive index strategy:

```bash
uv sync --index-strategy unsafe-best-match
```

If you do not use `uv`, create and activate a Python virtual environment and install dependencies declared in `pyproject.toml` using your preferred packaging tool (pip, pdm, poetry, etc.). Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install XXX XXX XXX  # replace with the actual dependencies
```

## Configuration

A YAML configuration file centralizes filesystem paths, hyperparameters, and runtime options. The default configuration file is `config/default.yml`.

- The CLI entry point `main.py` accepts a `--config` argument to specify an alternative configuration file. Example:

```bash
python main.py --config config/default.yml <step>
# or with uv
uv run main.py --config config/default.yml <step>
```

- To customize behavior, copy `config/default.yml` to a local file (for example `config/local.yml`) and modify that file.

- The configuration contains keys for common directories (data, raw data, models, results), file templates, and model hyperparameters (for the autoencoder, UMAP, and player training). See `config/default.yml` for full details.

## Project layout (high level)

- `main.py` — CLI entry point for orchestrating discrete pipeline stages.
- `config/` — YAML configuration files (`default.yml` and potential local overrides).
- `src/` — Source code: data ingestion, preprocessing, features, training, evaluation, visualization, UI.
- `data/` — Expected location for raw and processed datasets (configurable).
- `models/` — Saved model artifacts (configurable).
- `results/` — Generated evaluation results, figures, and tables (configurable).

## Quickstart (end-to-end minimal example)

Below is a sequence of commands that implements a representative end-to-end workflow. Replace `uv run` with `python` if you prefer to run without `uv`.

1. Acquire raw game data (fetch games from configured sources):

```bash
uv run main.py fetch
```

2. Build the processed dataset (parquet files and canonical columns):

```bash
uv run main.py build
```

3. Extract summary statistics (openings and per-player summaries):

```bash
uv run main.py stats
```

4. Compute position vector representations (feature extraction):

```bash
uv run main.py vectors
```

5. Train the autoencoder to compress board representations:

```bash
uv run main.py autoencoder
```

6. Fit and apply UMAP to the autoencoder latent representations:

```bash
# If cuda is available
uv run python -m cuml.accel main.py umap
# Else
uv run main.py umap
```

7. Compute Jensen-Shannon divergence (JSD)-based metrics and generate evaluation artifacts:

```bash
uv run main.py evaluate
```

8. Train champion-specific player embeddings (Maia variants):

```bash
uv run main.py train_players
```

9. Evaluate trained player models against held-out data:

```bash
# If cuda is available
uv run python -m cuml.accel evaluate_players
# Else
uv run main.py evaluate_players
```

10. Perform an MCTS parameter grid search and analyze results:

```bash
uv run main.py evaluate_mcts_params
uv run main.py generate_mcts_heatmaps
```

11. Simulate tournaments among champion players and baseline engines (supported formats: `single_elimination`, `round_robin`, `swiss_system`):

```bash
uv run main.py tournament --tournament round_robin
```

12. Compute final aggregated results and re-run model-specific AE→UMAP→JSD analyses:

```bash
uv run main.py results
```

13. Launch the interactive web interface:

```bash
uv run main.py ui
```

14. Generate all visualizations and tables used in the manuscript:

```bash
uv run main.py visualize
```

## Detailed explanation of pipeline stages

Each pipeline stage corresponds to a module under `src/` and emits logs to facilitate reproducibility. The mapping between `main.py` CLI steps and their implementations is as follows (the function signatures are provided as guidance):

- `fetch` — `src.data.fetch_games.fetch_all_games(config)`
  - Acquire external game sources and store raw files under the configured `data/raw/` path.

- `build` — `src.data.build_dataset.build_dataset(config)`
  - Parse raw files and construct canonical processed datasets (Parquet files) with the columns described in `config/default.yml`.

- `stats` — `src.data.opening_stats.extract_opening_stats(config)` and `src.data.players_stats.extract_players_stats(config)`
  - Produce aggregate statistics used for analysis and visualization.

- `vectors` — `src.features.compute_vectors.compute_vectors(config)`
  - Convert board states into numerical vectors appropriate for neural encoders.

- `autoencoder` — `src.training.train_autoencoder.run_autoencoder_pipeline(config)`
  - Train the autoencoder and write a checkpoint (default location: `models/saved/autoencoder.pth`).

- `umap` — `src.training.train_umap.run_umap_pipeline(config)`
  - Train or load a UMAP model and transform encoded vectors into a low-dimensional embedding.

- `evaluate` — `src.evaluation.compute_distances.*`
  - Compute JSD-based pairwise and cross-matrix distances used to quantify behavioral similarity.

- `train_players` — `src.training.train_players.run_training(config)`
  - Train champion-specific embeddings and save model checkpoints.

- `evaluate_players` — `src.evaluation.evaluate_players.evaluate_players(config, force_train=False)`
  - Evaluate player models on held-out datasets and produce comparative metrics.

- `evaluate_mcts_params` / `generate_mcts_heatmaps` — grid-search and visualization helpers for MCTS parameter analysis.

- `tournament` — `src.evaluation.tournament` classes and `src.models.maia.MaiaEngine`
  - Simulate matches and record tournament outcomes.

- `results` — utilities that compute final metrics and re-run the AE→UMAP→JSD pipeline for model variants.

- `ui` — `src.ui.app.run_ui(config)`
  - Start the web interface for interactive exploration of models and embeddings.

- `visualize` — visualization and table-building helpers under `src.visualization`.

## Outputs and reproducibility

All artifacts (models, processed datasets, evaluation tables, and figures) are written to the directories specified in `config/default.yml` (by default `data/`, `models/`, and `results/`). To reproduce experiments, record the following for each run:

- The full configuration file used (`--config` path).
- The exact command line invocation(s).
- The random seed(s) used by training and evaluation routines.
- The environment and package versions.

## Troubleshooting

- Dependency resolution with `uv` may fail; if so, use the permissive index strategy shown above.
- If a pipeline stage fails, inspect the stdout logs and any log files; the pipeline emits informative messages for missing files or misconfiguration.
