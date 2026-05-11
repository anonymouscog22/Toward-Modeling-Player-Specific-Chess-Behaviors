"""Training utilities for per-player style embeddings using the Maia backbone.

This module provides a dataset wrapper and a training routine intended to learn
per-player style embeddings that complement Maia's canonical Elo-category
embeddings. The learned per-player embeddings are initialised from Maia's most
representative Elo vector and are trained while preserving the original Maia
Elo embeddings as fixed (non-trainable) parameters.

The principal entry point is `run_training(config)`, which constructs a
`PlayerDataset`, configures the Maia model for per-player embedding training,
and persists the trained embeddings to disk along with learning curves.
"""

from typing import Dict

import chess
import polars as pl
import torch
import torch.nn as nn
from maia2 import model
from maia2.utils import (
    board_to_tensor,
    create_elo_dict,
    get_all_possible_moves,
    map_to_category,
    mirror_move,
)
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.player_style import PlayerStyleEmbedding

logger = getLogger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlayerDataset(Dataset):
    """Dataset providing board tensors and style indices for per-player training."""

    def __init__(
        self,
        data_path: str,
        player_dict: Dict[str, str],
        all_moves_dict: Dict[str, int],
    ):
        self.df = pl.read_parquet(data_path)
        self.player_dict = player_dict
        self.all_moves_dict = all_moves_dict
        self.elo_dict = create_elo_dict()
        self.max_maia_idx = max(self.elo_dict.values())

        self.player_to_idx = {
            player: idx + self.max_maia_idx + 1
            for idx, player in enumerate(player_dict.values())
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.row(idx, named=True)
        board = chess.Board(row["fen"])
        move_uci = row["move"]

        if row["player_color"] == "black":
            board = board.mirror()
            move_uci = mirror_move(move_uci)

        board_tensor = board_to_tensor(board)
        active_player = row["player_name"]
        opponent_elo = 2500

        if active_player in self.player_to_idx:
            active_player_idx = self.player_to_idx[active_player]
        else:
            active_player_idx = map_to_category(2500, self.elo_dict)

        opponent_idx = map_to_category(opponent_elo, self.elo_dict)
        move_label = self.all_moves_dict[move_uci]

        return board_tensor, active_player_idx, opponent_idx, move_label


def run_training(config: Config) -> None:
    """Train project-specific per-player embeddings using a frozen Maia backbone."""
    epochs = config.player_training.epochs
    batch_size = config.player_training.batch_size
    lr = config.player_training.learning_rate

    maia_model = model.from_pretrained("rapid", DEVICE)
    n_players = len(config.data.players)
    maia_model.elo_embedding = PlayerStyleEmbedding(
        maia_model.elo_embedding, n_players
    ).to(DEVICE)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}

    # Load training and test datasets
    logger.info("Loading datasets for training and evaluation (train and test)...")
    train_dataset = PlayerDataset(
        config.paths.train_set_path, config.data.players, all_moves_dict
    )
    test_dataset = PlayerDataset(
        config.paths.test_set_path, config.data.players, all_moves_dict
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    # Use a larger batch size for evaluation to reduce gradient-storage overhead and improve throughput
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4
    )

    maia_model.requires_grad_(False)
    maia_model.elo_embedding.players_embeddings.weight.requires_grad = True

    optimizer = Adam(maia_model.elo_embedding.players_embeddings.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(
        f"Commencing per-player embedding training for {epochs} epochs, batch_size={batch_size}, lr={lr}"
    )

    # Containers for training metrics and evaluation results
    history_train_loss = []
    history_train_acc = []
    history_test_acc = []

    pbar_epochs = tqdm(range(epochs), desc="Epochs", unit="epoch")

    for epoch in pbar_epochs:
        # ==========================================================
        # 1. TRAINING PHASE
        # ==========================================================
        maia_model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar_batches = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} [Train]",
            leave=False,
            unit="batch",
        )

        for boards, active_ids, opponent_ids, labels in pbar_batches:
            boards, active_ids, opponent_ids, labels = (
                boards.to(DEVICE, non_blocking=True),
                active_ids.to(DEVICE, non_blocking=True),
                opponent_ids.to(DEVICE, non_blocking=True),
                labels.to(DEVICE, non_blocking=True),
            )

            logits_maia, _, _ = maia_model(boards, active_ids, opponent_ids)
            loss = criterion(logits_maia, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics: Loss
            current_loss = loss.item()
            epoch_loss += current_loss

            # Metrics: Training accuracy accumulation
            preds = logits_maia.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar_batches.set_postfix({"batch_loss": f"{current_loss:.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        train_acc = train_correct / train_total

        # ==========================================================
        # 2. EVALUATION PHASE (TEST SET)
        # ==========================================================
        maia_model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for boards, active_ids, opponent_ids, labels in test_loader:
                boards, active_ids, opponent_ids, labels = (
                    boards.to(DEVICE, non_blocking=True),
                    active_ids.to(DEVICE, non_blocking=True),
                    opponent_ids.to(DEVICE, non_blocking=True),
                    labels.to(DEVICE, non_blocking=True),
                )

                logits_maia, _, _ = maia_model(boards, active_ids, opponent_ids)
                preds = logits_maia.argmax(dim=-1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        test_acc = test_correct / test_total

        # ==========================================================
        # 3. PERSIST METRICS
        # ==========================================================
        history_train_loss.append(avg_loss)
        history_train_acc.append(train_acc)
        history_test_acc.append(test_acc)

        pbar_epochs.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "test_acc": f"{test_acc:.4f}",
            }
        )

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}"
        )

    # ==========================================================
    # 4. PERSIST FINAL RESULTS
    # ==========================================================
    # Save learned per-player embeddings
    torch.save(
        maia_model.elo_embedding.players_embeddings.state_dict(),
        config.paths.champions_embeddings_path,
    )
    logger.info(
        "Player embedding model saved to %s", config.paths.champions_embeddings_path
    )

    # Persist training history to a Parquet file for later analysis
    df_history = pl.DataFrame(
        {
            "epoch": list(range(1, epochs + 1)),
            "train_loss": history_train_loss,
            "train_accuracy": history_train_acc,
            "test_accuracy": history_test_acc,
        }
    )

    history_path = config.paths.learning_curves_path
    df_history.write_parquet(str(history_path))
    logger.info("Training history persisted to %s", history_path)
