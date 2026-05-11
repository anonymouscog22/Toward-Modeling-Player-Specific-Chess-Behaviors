"""Module providing a wrapper around the Maia model for style-conditioned move prediction.

This module implements a `MaiaEngine` class which initializes a pre-trained Maia
network, augments it with a per-player style embedding, and exposes convenience
methods for single-move prediction, MCTS-based search, and batch evaluation.
"""

import io
import os

import chess
import chess.pgn
import numpy as np
import torch
from maia2 import inference, model
from maia2.utils import board_to_tensor, mirror_move
from tqdm import tqdm

from src.core.config import Config
from src.core.utils import getLogger
from src.models.mcts import MCTS
from src.models.player_style import PlayerStyleEmbedding

logger = getLogger()


class MaiaEngine:
    """A wrapper around the Maia model for style-aware move prediction and evaluation.

    The `MaiaEngine` encapsulates a pre-trained Maia neural network, attaches a
    learnable per-player style embedding, and provides:
    - `predict_move`: single-position move probability prediction and evaluation,
    - `predict_mcts`: a lightweight MCTS wrapper that uses `predict_move` as the
      child generator,
    - `evaluate_batch`: batched accuracy evaluation over a dataset.

    Parameters
    ----------
    config : Config
        Application configuration providing player lists and filesystem paths.
    model_type : str, optional
        Identifier of the pre-trained Maia backbone to load (default: "rapid").
    """

    def __init__(self, config: Config, model_type="rapid"):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.from_pretrained(model_type, self.device)
        self.prepare = inference.prepare()

        n_players = len(self.config.data.players)
        self.model.elo_embedding = PlayerStyleEmbedding(
            self.model.elo_embedding, n_players
        ).to(self.device)

        embed_path = self.config.paths.champions_embeddings_path
        if os.path.exists(embed_path):
            state_dict = torch.load(embed_path, map_location=self.device)
            self.model.elo_embedding.players_embeddings.load_state_dict(state_dict)
            logger.info(f"Player embeddings loaded from {embed_path}.")
        else:
            logger.warning(f"Player embeddings not found at {embed_path}.")

        self.player_to_idx = {
            player_name: idx + self.model.elo_embedding.max_maia_idx + 1
            for idx, player_name in enumerate(self.config.data.players.values())
        }

    def get_board_from_fen(self, fen, pgn):
        """Reconstruct a chess.Board from either a FEN string or a PGN text.

        If a non-empty PGN string is supplied and is parsable, the board is
        constructed by applying the moves from the PGN. If parsing fails or no
        PGN is supplied, the function falls back to creating the board from the
        provided FEN string.

        Parameters
        ----------
        fen : str
            Forsyth–Edwards Notation string describing the board state.
        pgn : str
            Optional PGN text containing the full game moves.

        Returns
        -------
        chess.Board
            A board object representing the position obtained from the PGN or FEN.
        """
        board = chess.Board()
        if pgn != "":
            try:
                pgn_io = io.StringIO(pgn)
                game = chess.pgn.read_game(pgn_io)
            except Exception:
                game = None
            if game is not None:
                for move in game.mainline_moves():
                    board.push(move)
            else:
                board = chess.Board(fen)
        else:
            board = chess.Board(fen)
        return board

    def predict_mcts(
        self,
        fen,
        pgn,
        num_simulations=1000,
        c_puct=1.5,
        threshold=0.01,
        active_elo: int | str = 2500,
        opponent_elo: int | str = 2500,
    ):
        """Perform a lightweight MCTS search using `predict_move` as the child generator.

        The method reconstructs the current board from FEN/PGN, initializes an
        MCTS instance that delegates child evaluation to `predict_move`, and
        returns the MCTS-selected root move together with the move-probability
        dictionary produced at the root.

        Parameters
        ----------
        fen : str
            FEN representation of the current board position.
        pgn : str
            Optional PGN text that can be used to reconstruct the board.
        num_simulations : int
            Number of MCTS simulations to run.
        c_puct : float
            Exploration constant used in the UCT-like selection formula.
        threshold : float
            Probability threshold below which child moves are pruned.
        active_elo : int | str
            Style identifier or Elo value for the active player.
        opponent_elo : int | str
            Style identifier or Elo value for the opponent.

        Returns
        -------
        tuple
            A pair (best_move, result) where `best_move` is the move selected at
            the root and `result` is a dict mapping candidate moves to model
            probabilities.
        """
        board = self.get_board_from_fen(fen, pgn)
        mcts = MCTS(self.predict_move)
        best_move, result = mcts.run(
            board,
            num_simulations,
            threshold=threshold,
            c_puct=c_puct,
            activ_elo=active_elo,
            opp_elo=opponent_elo,
        )

        return best_move, result

    def _get_style_idx(self, val: int | str):
        """Map a provided style identifier or Elo value to an internal Maia index.

        The function accepts either a string identifier (which may refer to a
        custom player name or a canonical Maia category) or an integer Elo
        value and returns the corresponding integer index used by the Maia model.
        """
        if isinstance(val, str) and val in self.player_to_idx:
            return self.player_to_idx[val]

        _, elo_dict, _ = self.prepare
        if isinstance(val, str) and val in elo_dict:
            return elo_dict[val]

        return inference.map_to_category(int(val), elo_dict)

    def predict_move(
        self, fen, active_elo: int | str = 2500, opponent_elo: int | str = 2500
    ):
        """Predict move probabilities and a scalar evaluation for a single position.

        The method constructs a tensor representation of the board (mirroring
        if Black is to move), conditions the Maia model on style indices for the
        active and opponent players, and returns a sorted dictionary of legal
        moves with their associated probabilities as well as the scalar value
        predicted by the Maia value head.

        Parameters
        ----------
        fen : str
            FEN string representing the position to evaluate.
        active_elo : int | str
            Style identifier or Elo value for the active player.
        opponent_elo : int | str
            Style identifier or Elo value for the opponent.

        Returns
        -------
        tuple
            A triple (best_move, move_probabilities, board_value) where
            `move_probabilities` is an ordered mapping from UCI moves to probabilities
            and `board_value` is the scalar evaluation returned by Maia.
        """
        board = chess.Board(fen)
        is_mirrored = False
        if board.turn == chess.BLACK:
            board = board.mirror()
            is_mirrored = True

        device = self.device
        board_tensor = board_to_tensor(board).unsqueeze(0).to(device)
        s_self = torch.tensor([self._get_style_idx(active_elo)]).to(device)
        s_oppo = torch.tensor([self._get_style_idx(opponent_elo)]).to(device)

        self.model.eval()
        with torch.no_grad():
            logits_maia, _, value_maia = self.model(board_tensor, s_self, s_oppo)
            board_value = float(value_maia[0].cpu().item())

            all_moves_dict, _, all_moves_dict_reversed = self.prepare
            legal_mask = torch.zeros(logits_maia.size(-1)).to(device)
            for move in board.legal_moves:
                legal_mask[all_moves_dict[move.uci()]] = 1

            probs = (logits_maia[0] * legal_mask).softmax(dim=-1).cpu().numpy()

        move_probs = {}
        for i in legal_mask.nonzero().flatten().tolist():
            move_uci = all_moves_dict_reversed[i]
            final_move = mirror_move(move_uci) if is_mirrored else move_uci
            move_probs[final_move] = float(probs[i])

        sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
        best_move = sorted_moves[0][0]

        return best_move, dict(sorted_moves), board_value

    def evaluate_batch(self, dataloader):
        """Evaluate batch-wise move-prediction accuracy for the current Maia model.

        This routine performs batched inference over the supplied dataloader and
        returns two concatenated numpy arrays: a boolean mask indicating whether
        each prediction was correct, and the corresponding active player indices.
        """
        self.model.eval()
        all_correct_preds = []
        all_player_ids = []

        with torch.no_grad():
            for boards, active_ids, opponent_ids, labels, legal_masks in tqdm(
                dataloader, desc="Batch evaluation"
            ):
                boards = boards.to(self.device, non_blocking=True)
                active_ids = active_ids.to(self.device, non_blocking=True)
                opponent_ids = opponent_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                legal_masks = legal_masks.to(self.device, non_blocking=True)

                logits, _, _ = self.model(boards, active_ids, opponent_ids)
                logits = logits.masked_fill(~legal_masks, -float("inf"))
                predictions = logits.argmax(dim=-1)

                all_correct_preds.append((predictions == labels).cpu().numpy())
                all_player_ids.append(active_ids.cpu().numpy())

        return np.concatenate(all_correct_preds), np.concatenate(all_player_ids)
