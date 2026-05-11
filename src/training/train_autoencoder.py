"""Training utilities for the autoencoder model.

This module implements training and inference routines for a feed-forward
autoencoder that compresses fixed-length position vectors. It exposes
functions to train the model on the training vectors and to encode datasets
(separately for train and test) into latent representations.
"""

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.autoencoder import Autoencoder

logger = getLogger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ChessDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.float32)


def train_autoencoder(config: Config) -> Autoencoder:
    """Train the autoencoder exclusively on the training dataset.

    The function loads precomputed training vectors from disk, constructs an
    Autoencoder instance, and trains it using hyperparameters specified in the
    configuration object. Only the training split is used for weight updates to
    ensure a clear separation between training and evaluation data.
    """

    data = np.load(config.paths.train_vectors_path, mmap_mode="r")
    input_dim = data.shape[1]

    model = Autoencoder(input_dim, config.autoencoder.latent_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=config.autoencoder.learning_rate)

    train_loader = DataLoader(
        ChessDataset(data),
        batch_size=config.autoencoder.batch_size,
        shuffle=True,
        num_workers=config.autoencoder.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    logger.info(f"Commencing training on {DEVICE} (training set only)")

    for epoch in range(config.autoencoder.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.autoencoder.epochs}",
            unit="batch",
        )

        for batch in pbar:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = (
            epoch_loss / len(train_loader) if len(train_loader) > 0 else float("nan")
        )
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config.paths.autoencoder_model_path)
    logger.info(f"Model saved to {config.paths.autoencoder_model_path}")
    return model


def _infer_and_save(
    model: Autoencoder, input_path: str, output_path: str, batch_size: int, desc: str
):
    """Utility to encode a dataset and persist latent vectors.

    Loads input vectors from `input_path`, encodes them using `model.encode`
    in batches, concatenates the resulting latent arrays and saves them to
    `output_path`.
    """
    data = np.load(input_path, mmap_mode="r")
    loader = DataLoader(ChessDataset(data), batch_size=batch_size, shuffle=False)
    encoded_vectors = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Encoding ({desc})", unit="batch"):
            batch = batch.to(DEVICE)
            latent = model.encode(batch)
            encoded_vectors.append(latent.cpu().numpy())

    result = np.concatenate(encoded_vectors, axis=0)
    np.save(output_path, result)
    logger.info(f"[{desc}] Latent vectors saved to {output_path}")


def infer_autoencoder(config: Config):
    """Encode the training and test datasets into latent representations.

    The function instantiates the model, loads the trained state dictionary,
    and performs batched encoding of both the training and test vector sets.
    Resulting latent arrays are saved to the filesystem paths defined in the
    configuration.
    """

    sample_data = np.load(config.paths.train_vectors_path, mmap_mode="r")
    input_dim = sample_data.shape[1]

    model = Autoencoder(input_dim, config.autoencoder.latent_dim).to(DEVICE)
    model.load_state_dict(
        torch.load(config.paths.autoencoder_model_path, map_location=DEVICE)
    )
    model.eval()

    logger.info("Commencing inference on the training and test datasets...")

    _infer_and_save(
        model,
        config.paths.train_vectors_path,
        config.paths.train_encoded_vectors_path,
        config.autoencoder.batch_size,
        "TRAIN",
    )

    _infer_and_save(
        model,
        config.paths.test_vectors_path,
        config.paths.test_encoded_vectors_path,
        config.autoencoder.batch_size,
        "TEST",
    )


def run_autoencoder_pipeline(config: Config):
    """Execute training followed by encoding inference for train and test.

    This convenience function runs the complete autoencoder workflow: it first
    performs model training using the precomputed training vectors, and then
    executes batched inference to produce latent representations for both the
    training and test datasets. Persisted model weights and encoded outputs are
    written to the filesystem paths defined in the configuration object.
    """
    train_autoencoder(config)
    infer_autoencoder(config)
