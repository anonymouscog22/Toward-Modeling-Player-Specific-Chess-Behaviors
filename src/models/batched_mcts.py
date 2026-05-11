"""Batched Monte Carlo Tree Search for GPU saturation.

This module implements a Batched MCTS that traverses multiple game trees
simultaneously, gathers their leaf nodes, and evaluates them in a single
batched GPU inference step via the Maia engine.
"""

from typing import Dict, List, Tuple

import chess
import numpy as np
import torch
from maia2.utils import board_to_tensor, mirror_move


class BatchedNode:
    """MCTS node for the batched implementation."""

    def __init__(self, maia_prob: float = 1.0) -> None:
        self.maia_prob: float = maia_prob
        self.children: Dict[str, "BatchedNode"] = {}
        self.visits: int = 0
        self.value: float = 0.0

    def compute_Q(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value / self.visits

    def compute_U(self, parent_visits: int, c_puct: float = 1.0) -> float:
        return c_puct * self.maia_prob * np.sqrt(parent_visits) / (1 + self.visits)


class BatchedMCTSManager:
    """Manages multiple MCTS trees to evaluate leaf nodes in batches."""

    def __init__(self, engine, c_puct: float = 1.5, threshold: float = 0.01) -> None:
        self.engine = engine
        self.c_puct = c_puct
        self.threshold = threshold
        self.device = engine.device
        self.all_moves_dict, _, self.all_moves_dict_reversed = engine.prepare

    def _predict_batch(
        self, boards: List[chess.Board], active_elos: List[str]
    ) -> Tuple[List[Dict[str, float]], np.ndarray]:
        """Evaluates a batch of board positions in a single network pass."""
        tensors = []
        is_mirrored_list = []
        process_boards = []  # New list to store correctly oriented boards for inference

        for b in boards:
            if b.turn == chess.BLACK:
                mb = b.mirror()
                is_mirrored_list.append(True)
                process_boards.append(mb)
                tensors.append(board_to_tensor(mb))
            else:
                is_mirrored_list.append(False)
                process_boards.append(b)
                tensors.append(board_to_tensor(b))

        batch_tensor = torch.stack(tensors).to(self.device, non_blocking=True)
        s_self = torch.tensor(
            [self.engine._get_style_idx(elo) for elo in active_elos]
        ).to(self.device, non_blocking=True)
        s_oppo = torch.tensor(
            [self.engine._get_style_idx(2500) for _ in active_elos]
        ).to(self.device, non_blocking=True)

        self.engine.model.eval()
        with torch.no_grad():
            logits, _, values = self.engine.model(batch_tensor, s_self, s_oppo)
            values = values.cpu().numpy().flatten()

            # Mask legal moves using the oriented boards stored in `process_boards`
            legal_masks = torch.zeros_like(logits, dtype=torch.bool).to(self.device)
            for i, pb in enumerate(process_boards):
                for move in pb.legal_moves:
                    legal_masks[i, self.all_moves_dict[move.uci()]] = True

            # Apply masked_fill to invalidate illegal move logits
            logits = logits.masked_fill(~legal_masks, -float("inf"))
            probs = logits.softmax(dim=-1).cpu().numpy()
            legal_masks_cpu = legal_masks.cpu().numpy()

        probs_list = []
        for i in range(len(boards)):
            move_probs = {}
            legal_indices = legal_masks_cpu[i].nonzero()[0]
            for idx in legal_indices:
                move_uci = self.all_moves_dict_reversed[idx]
                final_move = mirror_move(move_uci) if is_mirrored_list[i] else move_uci
                move_probs[final_move] = float(probs[i, idx])
            probs_list.append(move_probs)

        return probs_list, values

    def run_batch(
        self,
        fens: List[str],
        active_elos: List[str],
        num_simulations: int = 100,
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """Runs the MCTS algorithm concurrently for a batch of initial positions."""
        batch_size = len(fens)
        roots = [BatchedNode() for _ in range(batch_size)]
        boards = [chess.Board(fen) for fen in fens]

        # 1. Initial expansion of the root with high-probability moves
        probs_list, _ = self._predict_batch(boards, active_elos)
        for i in range(batch_size):
            for move, prob in probs_list[i].items():
                if prob > self.threshold:
                    roots[i].children[move] = BatchedNode(prob)

        # 2. Simulation loop (MCTS iterations)
        for _ in range(num_simulations):
            paths = []
            leaf_boards = []
            leaf_indices = []
            terminal_values = {}

            # Phase A: Selection step within the tree policy
            for i in range(batch_size):
                node = roots[i]
                sim_board = boards[i].copy()
                path = [node]

                while node.children:
                    best_score = -float("inf")
                    best_move = ""
                    best_child = None

                    for move, child in node.children.items():
                        score = child.compute_Q() + child.compute_U(
                            node.visits, self.c_puct
                        )
                        if score > best_score:
                            best_score = score
                            best_move = move
                            best_child = child

                    if best_child is None:
                        break

                    sim_board.push_uci(best_move)
                    node = best_child
                    path.append(node)

                paths.append(path)

                if sim_board.is_game_over():
                    outcome = sim_board.outcome()
                    if outcome is None or outcome.winner is None:
                        terminal_values[i] = 0.0
                    else:
                        terminal_values[i] = -1.0
                else:
                    leaf_boards.append(sim_board)
                    leaf_indices.append(i)

            # Phase B: Batch evaluation of leaf nodes (network inference)
            if leaf_boards:
                leaf_elos = [active_elos[idx] for idx in leaf_indices]
                probs_list, values = self._predict_batch(leaf_boards, leaf_elos)

                # Expansion
                for k, idx in enumerate(leaf_indices):
                    node = paths[idx][-1]
                    for move, prob in probs_list[k].items():
                        if prob > self.threshold:
                            node.children[move] = BatchedNode(prob)
            else:
                values = []

            # Phase C: Backpropagation of values through the traversed paths
            val_idx = 0
            for i in range(batch_size):
                if i in terminal_values:
                    val = terminal_values[i]
                else:
                    val = float(values[val_idx])
                    val_idx += 1

                for node in reversed(paths[i]):
                    node.visits += 1
                    node.value += val
                    val = -val  # Alternate player perspective during backpropagation

        # 3. Extract final results and return best moves + probability distributions
        best_root_moves = []
        root_probs_list = []

        for i in range(batch_size):
            if not roots[i].children:
                best_root_moves.append(None)
                root_probs_list.append({})
                continue

            moves = list(roots[i].children.keys())
            visit_counts = np.array(
                [child.visits for child in roots[i].children.values()], dtype=np.float64
            )

            if temperature == 0:
                best_move = moves[np.argmax(visit_counts)]
            else:
                visits_with_temp = np.power(visit_counts, 1.0 / temperature)
                probabilities = visits_with_temp / np.sum(visits_with_temp)
                best_move = np.random.choice(moves, p=probabilities)

            result_probs = {
                move: child.maia_prob for move, child in roots[i].children.items()
            }
            best_root_moves.append(best_move)
            root_probs_list.append(result_probs)

        return best_root_moves, root_probs_list
