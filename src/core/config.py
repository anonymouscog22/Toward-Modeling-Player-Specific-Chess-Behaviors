"""Application configuration models and filesystem helpers.

This module centralizes project-wide configuration using Pydantic models. It
defines structured containers for filesystem paths, data acquisition settings,
and modelling hyperparameters. Helper methods are provided to create the
required directory layout and to load configuration overrides from a YAML file.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Filesystem layout and canonical paths used by the project.

    Attributes
    ----------
    data, raw_data, model, result, evaluation_dir : str
        Base directories used to organize the project's artifacts.
    dataset_path, train_set_path, test_set_path, opening_stats_path : str
        Canonical locations for processed dataset artifacts.
    train_vectors_path, test_vectors_path : str
        Paths to the NumPy arrays containing position vectors.
    autoencoder_model_path : str
        Location where the trained autoencoder state dict is persisted.
    train_encoded_vectors_path, test_encoded_vectors_path : str
        Paths for latent vectors produced by the autoencoder.
    train_umap_result_path, test_umap_result_path, umap_model_path : str
        Paths for UMAP outputs and serialized model.
    """

    data: str = "data/"
    raw_data: str = "data/raw/"
    model: str = "models/"
    result: str = "results/"
    evaluation_dir: str = "results/evaluation/"

    dataset_path: str = "data/processed/dataset.parquet"
    train_set_path: str = "data/processed/train.parquet"
    test_set_path: str = "data/processed/test.parquet"
    opening_stats_path: str = "data/processed/opening_stats.parquet"
    player_stats_path: str = "data/processed/player_stats.parquet"

    train_vectors_path: str = "data/processed/train_vectors.npy"
    test_vectors_path: str = "data/processed/test_vectors.npy"

    autoencoder_model_path: str = "models/saved/autoencoder.pth"

    train_encoded_vectors_path: str = "data/processed/train_encoded_vectors.npy"
    test_encoded_vectors_path: str = "data/processed/test_encoded_vectors.npy"

    train_umap_result_path: str = "data/processed/train_umap.parquet"
    test_umap_result_path: str = "data/processed/test_umap.parquet"
    umap_model_path: str = "models/saved/style_umap.pkl"

    champions_embeddings_path: str = "models/saved/champions_embeddings.pth"
    learning_curves_path: str = "results/evaluation/learning_curves.parquet"
    player_accuracies_path: str = (
        "results/evaluation/player_accuracies_comparison.parquet"
    )
    predictions_path: str = "results/evaluation/predictions.parquet"
    accuracy_path: str = "results/evaluation/accuracy.parquet"
    accuracy_table_latex_path: str = "results/graphics/accuracy_table.tex"

    # Graphics paths (per-method variants will be constructed from these base names)
    moves_distribution_graph_path: str = "results/graphics/moves_distribution.pdf"
    jsd_heatmap_path: str = "results/graphics/jsd_heatmap.pdf"
    jsd_stability_heatmap_path: str = "results/graphics/jsd_stability_heatmap.pdf"

    # New configurable templates for method-specific outputs and embeddings
    method_jsd_heatmap_template: str = "results/graphics/jsd_heatmap_{method}.pdf"
    method_jsd_stability_template: str = (
        "results/graphics/jsd_stability_heatmap_{method}.pdf"
    )
    # Template for a swapped stability heatmap where rows=Real(Test) and cols=Predicted(Model)
    method_jsd_stability_real_pred_template: str = (
        "results/graphics/jsd_stability_heatmap_{method}_real_vs_pred.pdf"
    )
    method_train_embeddings_template: str = "data/processed/train_{method}.parquet"
    method_test_embeddings_template: str = "data/processed/test_{method}.parquet"

    table_latex_path: str = "results/graphics/dataset_table.tex"
    ae_table_latex_path: str = "results/graphics/autoencoder_table.tex"
    jsd_table_latex_path: str = "results/graphics/jsd_stability_table.tex"

    def make_directories(self) -> None:
        """Ensure all configured filesystem paths exist on disk.

        For each configured path, this helper creates parent directories as
        required. If a configured value represents a file (i.e., has a suffix),
        its parent directory is created; otherwise the path itself is treated
        as a directory and created.

        This method is idempotent and safe to call multiple times.
        """
        for path_str in self.model_dump().values():
            path_obj = Path(path_str)

            if path_obj.suffix:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            else:
                path_obj.mkdir(parents=True, exist_ok=True)

    def get_embeddings_path(self, method: str, is_test: bool) -> str:
        """Return a canonical path for embeddings produced by `method`.

        Parameters
        ----------
        method : str
            Short identifier of the embedding method (e.g., 'umap', 'pca').
        is_test : bool
            Whether the requested path is for the test split.

        Returns
        -------
        str
            Filesystem path where embeddings for the requested split and method
            should be stored.
        """
        split = "test" if is_test else "train"
        return f"data/processed/{split}_{method}.parquet"

    def get_distances_path(self, method: str, is_test: bool, kde: bool) -> str:
        """Return a canonical path for distance tables for a given method."""
        split = "test" if is_test else "train"
        kde_suffix = "_kde" if kde else ""
        return f"{self.evaluation_dir}distances_{split}_{method}{kde_suffix}.parquet"

    def get_cross_distances_path(self, method: str, kde: bool) -> str:
        """Return the canonical path for cross-split distances for `method`."""
        kde_suffix = "_kde" if kde else ""
        return f"{self.evaluation_dir}cross_distances_{method}{kde_suffix}.parquet"

    def get_full_cross_matrix_path(self, method: str, kde: bool) -> str:
        kde_suffix = "_kde" if kde else ""
        return f"{self.evaluation_dir}full_cross_distances_{method}{kde_suffix}.parquet"


class JSDConfig(BaseModel):
    method: str = "umap"
    kde: bool = False


class PlayerTrainingConfig(BaseModel):
    """Per-player training hyperparameters.

    These settings may be used when performing player-specific training or
    fine-tuning routines. Values provided here are intended as sensible
    defaults and may be overridden via a YAML configuration file.

    Attributes
    ----------
    epochs : int
        Number of epochs to train for.
    batch_size : int
        Mini-batch size used during training.
    learning_rate : float
        Learning rate used by the optimizer.
    """

    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-4


class UMAPConfig(BaseModel):
    """Configuration for UMAP dimensionality reduction."""

    n_components: int = 2


class AutoencoderConfig(BaseModel):
    """Hyperparameters for the autoencoder training routine."""

    latent_dim: int = 128
    epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 1e-3
    num_workers: int = 0


class DataConfig(BaseModel):
    """Data acquisition and dataset-related configuration.

    Attributes
    ----------
    max_workers : int
        Maximum number of worker threads used for concurrent downloads.
    headers : dict
        HTTP headers used for web requests.
    players : dict
        Mapping from remote player identifier to human-readable player name.
    dataset_col_order : list
        Preferred column ordering for the produced dataset DataFrame.
    """

    max_workers: int = 5
    headers: dict = Field(
        default_factory=lambda: {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://www.google.com/",
        }
    )
    players: dict = Field(
        default_factory=lambda: {
            "10240": "Alekhine",
            "12112": "Andersson",
            "12088": "Anand",
            "13755": "Beliavsky",
            "47544": "Capablanca",
            "19233": "Fischer",
            "12183": "Ivanchuk",
            "20719": "Karpov",
            "15940": "Kasparov",
            "15866": "Korchnoi",
            "11227": "Larsen",
            "16149": "Petrosian",
            "14568": "Portisch",
            "12181": "Short",
            "14380": "Tal",
            "14220": "Timman",
        }
    )
    dataset_col_order: list = Field(
        default_factory=lambda: [
            "game_id",
            "round",
            "player_name",
            "player_color",
            "fen",
            "move",
            "repetition",
            "result",
        ]
    )


class Config(BaseModel):
    """Top-level application configuration model.

    Instances of this class aggregate `PathsConfig`, `DataConfig`,
    `AutoencoderConfig` and `UMAPConfig`. The class method `from_yaml` permits
    loading overrides from a YAML file; when invoked it also ensures the
    configured filesystem layout exists on disk.
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    autoencoder: AutoencoderConfig = Field(default_factory=AutoencoderConfig)
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    player_training: PlayerTrainingConfig = Field(default_factory=PlayerTrainingConfig)
    jsd: JSDConfig = Field(default_factory=JSDConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Instantiate a `Config`, optionally overriding defaults with a YAML file.

        If `yaml_path` does not exist, an instance populated with default values
        is returned. When a YAML file is present its keys are validated and used
        to construct the Pydantic model. After construction this helper ensures
        that filesystem locations required by `PathsConfig` exist by invoking
        `paths.make_directories()`.

        Parameters
        ----------
        yaml_path : str
            Filesystem path to a YAML configuration file.

        Returns
        -------
        Config
            A validated and fully-initialized configuration instance.
        """
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            yaml_dict = yaml.safe_load(f) or {}

        config_instance = cls(**yaml_dict)
        config_instance.paths.make_directories()
        return config_instance
