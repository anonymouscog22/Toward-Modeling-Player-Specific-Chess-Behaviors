"""Evaluation routines for assessing per-player predictive performance.

This module provides dataset utilities and an evaluation driver that compares
the predictive accuracy and Jensen-Shannon Divergence (JSD) of a Maia model
conditioned on learned per-player embeddings, against a baseline Maia model
and a Custom model augmented with Monte Carlo Tree Search (MCTS).
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import chess
import polars as pl
import torch
from maia2.model import from_pretrained
from maia2.utils import board_to_tensor, get_all_possible_moves, mirror_move
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.batched_mcts import BatchedMCTSManager
from src.models.maia import MaiaEngine
from src.training.train_players import run_training

logger = getLogger()


class EvaluationDataset(Dataset):
    """Dataset providing examples for evaluation of move-prediction accuracy.

    Each item is a tuple (board_tensor, active_player_idx, opponent_idx,
    move_label, legal_mask) suitable for model inference and accuracy
    computation. When the active player is Black the board and the move label
    are mirrored to maintain a White-to-move canonical representation.
    """

    def __init__(
        self,
        data_path: str,
        player_to_idx: Dict[str, int],
        all_moves_dict: Dict[str, int],
        base_elo_idx: int,
    ) -> None:
        self.df = pl.read_parquet(data_path)
        self.player_to_idx = player_to_idx
        self.all_moves_dict = all_moves_dict
        self.base_elo_idx = base_elo_idx
        self.num_moves = len(all_moves_dict)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int, torch.Tensor]:
        row = self.df.row(idx, named=True)
        board = chess.Board(row["fen"])
        move_uci = row["move"]

        if row["player_color"] == "black":
            board = board.mirror()
            move_uci = mirror_move(move_uci)

        board_tensor = board_to_tensor(board)

        legal_mask = torch.zeros(self.num_moves, dtype=torch.bool)
        for move in board.legal_moves:
            if move.uci() in self.all_moves_dict:
                legal_mask[self.all_moves_dict[move.uci()]] = True

        active_player = row["player_name"]
        active_player_idx = self.player_to_idx.get(active_player, self.base_elo_idx)
        opponent_idx = self.base_elo_idx
        move_label = self.all_moves_dict[move_uci]

        return board_tensor, active_player_idx, opponent_idx, move_label, legal_mask


def mcts_worker(
    fens_chunk, players_chunk, config, num_simulations, batch_size, worker_id
):
    """Process-local worker that executes batched MCTS on an input chunk.

    This function is intended to run in a separate process. It iterates over its
    assigned chunk of positions and uses a BatchedMCTSManager to compute
    move predictions and root-probability distributions.
    """

    engine = MaiaEngine(config)
    mcts_manager = BatchedMCTSManager(engine, threshold=0.01)

    all_best_moves = []
    all_probs = []

    # The progress bar is positioned per-worker via the `position` parameter.
    for i in tqdm(
        range(0, len(fens_chunk), batch_size),
        desc=f"Worker {worker_id + 1}",
        position=worker_id,
        leave=True,
    ):
        batch_fens = fens_chunk[i : i + batch_size]
        batch_players = players_chunk[i : i + batch_size]

        best_moves, root_probs_list = mcts_manager.run_batch(
            fens=batch_fens,
            active_elos=batch_players,
            num_simulations=num_simulations,
        )

        all_best_moves.extend(best_moves)
        all_probs.extend([json.dumps(p) for p in root_probs_list])

    return all_best_moves, all_probs


def generate_predictions_parquet(
    config: Config, num_mcts_simulations: int = 1000
) -> None:
    """Generates predictions and probabilities for Baseline, Custom, and MCTS models."""
    engine = MaiaEngine(config)
    baseline_model = from_pretrained("rapid", device=engine.device)
    baseline_model.eval()
    engine.model.eval()

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    all_moves_dict_reversed = {i: move for move, i in all_moves_dict.items()}

    _base_elo_idx = engine._get_style_idx(2500)
    base_elo_idx = int(_base_elo_idx) if _base_elo_idx is not None else 0

    df_test = pl.read_parquet(config.paths.test_set_path)

    test_dataset = EvaluationDataset(
        config.paths.test_set_path, engine.player_to_idx, all_moves_dict, base_elo_idx
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=4)

    baseline_preds_temp = []
    baseline_probs_temp = []
    custom_preds_temp = []
    custom_probs_temp = []

    # =========================================================================
    # 1. BATCH INFERENCE (Baseline & Custom Models)
    # =========================================================================
    logger.info("1/3 Performing batched inference for Baseline and Custom models...")
    with torch.no_grad():
        for boards, active_ids, opponent_ids, _, legal_masks in tqdm(
            test_loader, desc="Batch Inference"
        ):
            boards = boards.to(engine.device, non_blocking=True)
            active_ids = active_ids.to(engine.device, non_blocking=True)
            opponent_ids = opponent_ids.to(engine.device, non_blocking=True)
            legal_masks = legal_masks.to(engine.device, non_blocking=True)

            base_ids = torch.full_like(active_ids, base_elo_idx)

            # --- Baseline Model ---
            logits_base, _, _ = baseline_model(boards, base_ids, base_ids)
            logits_base = logits_base.masked_fill(~legal_masks, -float("inf"))
            probs_base = logits_base.softmax(dim=-1)
            preds_base = logits_base.argmax(dim=-1)

            # --- Custom Model ---
            logits_custom, _, _ = engine.model(boards, active_ids, base_ids)
            logits_custom = logits_custom.masked_fill(~legal_masks, -float("inf"))
            probs_custom = logits_custom.softmax(dim=-1)
            preds_custom = logits_custom.argmax(dim=-1)

            probs_base_cpu = probs_base.cpu().numpy()
            probs_custom_cpu = probs_custom.cpu().numpy()
            legal_masks_cpu = legal_masks.cpu().numpy()
            preds_base_cpu = preds_base.cpu().numpy()
            preds_custom_cpu = preds_custom.cpu().numpy()

            for i in range(boards.size(0)):
                dict_base = {}
                dict_custom = {}
                legal_indices = legal_masks_cpu[i].nonzero()[0]

                for idx in legal_indices:
                    move_uci = all_moves_dict_reversed[idx]
                    dict_base[move_uci] = float(probs_base_cpu[i, idx])
                    dict_custom[move_uci] = float(probs_custom_cpu[i, idx])

                baseline_probs_temp.append(dict_base)
                custom_probs_temp.append(dict_custom)
                baseline_preds_temp.append(all_moves_dict_reversed[preds_base_cpu[i]])
                custom_preds_temp.append(all_moves_dict_reversed[preds_custom_cpu[i]])

    # =========================================================================
    # 2. ALIGNMENT AND DEMIRRORING (Baseline & Custom Models)
    # =========================================================================
    logger.info("2/3 Aligning and demirroring move probability spaces...")
    colors = df_test["player_color"].to_list()

    final_base_preds, final_base_probs = [], []
    final_custom_preds, final_custom_probs = [], []

    for idx in range(len(colors)):
        is_black = colors[idx] == "black"

        p_base = baseline_probs_temp[idx]
        p_cust = custom_probs_temp[idx]
        pred_b = baseline_preds_temp[idx]
        pred_c = custom_preds_temp[idx]

        if is_black:
            p_base = {mirror_move(m): v for m, v in p_base.items()}
            p_cust = {mirror_move(m): v for m, v in p_cust.items()}
            pred_b = mirror_move(pred_b)
            pred_c = mirror_move(pred_c)

        final_base_preds.append(pred_b)
        final_base_probs.append(json.dumps(p_base))
        final_custom_preds.append(pred_c)
        final_custom_probs.append(json.dumps(p_cust))

    # =========================================================================
    # 3. MCTS INFERENCE (MULTI-PROCESSING + BATCHING)
    # =========================================================================
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    fens_list = df_test["fen"].to_list()
    players_list = df_test["player_name"].to_list()

    num_workers = 24
    worker_batch_size = 256

    # Partition the dataset and assign a worker identifier to each chunk
    chunk_size = math.ceil(len(fens_list) / num_workers)
    chunks = []
    w_id = 0
    for i in range(0, len(fens_list), chunk_size):
        chunks.append(
            (
                fens_list[i : i + chunk_size],
                players_list[i : i + chunk_size],
                w_id,
            )
        )
        w_id += 1

    mcts_preds = []
    mcts_probs = []

    logger.info(
        "3/3 Performing MCTS inference using %d worker processes...", num_workers
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for fens_chunk, players_chunk, worker_id in chunks:
            futures.append(
                executor.submit(
                    mcts_worker,
                    fens_chunk,
                    players_chunk,
                    config,
                    num_mcts_simulations,
                    worker_batch_size,
                    worker_id,
                )
            )

        for future in futures:
            b_moves, b_probs = future.result()
            mcts_preds.extend(b_moves)
            mcts_probs.extend(b_probs)

    # =========================================================================
    # 4. ASSEMBLY AND PERSISTENCE
    # =========================================================================
    df_predictions = df_test.select(["game_id", "fen", "player_name", "move"]).rename(
        {"move": "true_move"}
    )
    df_predictions = df_predictions.with_columns(
        [
            pl.Series("pred_baseline", final_base_preds),
            pl.Series("probs_baseline", final_base_probs),
            pl.Series("pred_custom", final_custom_preds),
            pl.Series("probs_custom", final_custom_probs),
            pl.Series("pred_mcts", mcts_preds),
            pl.Series("probs_mcts", mcts_probs),
        ]
    )

    output_path = config.paths.predictions_path
    df_predictions.write_parquet(output_path)
    logger.info("Predictions file has been saved to %s", output_path)


def evaluate_players(
    config: Config, force_train: bool = False, num_mcts_simulations: int = 100
) -> None:
    """Evaluate per-player predictive accuracy and JSD comparing Baseline, Custom and MCTS.

    This function optionally triggers per-player embedding training, generates a dataset
    of predictions (batching direct models, simulating MCTS), and finally computes and
    aggregates accuracy and JSD metrics.
    """
    if force_train:
        logger.info(
            "Forcing (re)training of per-player embeddings prior to evaluation..."
        )
        run_training(config)

    # Ensure evaluation directory exists and generate predictions parquet
    eval_dir = Path(config.paths.evaluation_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    generate_predictions_parquet(config, num_mcts_simulations)
