"""Utilities for converting chess positions to vectors and persisting UMAP instances.

This module provides a lightweight subclass of UMAP that exposes safe persistence
methods, alongside a function to transform a (FEN, move) pair into a flat tensor
suitable for downstream representation learning.
"""

import pickle
from pathlib import Path

import chess
import torch
from maia2.utils import board_to_tensor
from umap import UMAP


class StyleUMAP(UMAP):
    """A thin UMAP wrapper that adds convenient persistence helpers.

    The subclass preserves the full UMAP instance state using Python's pickle
    facility. These helpers centralize model save/load semantics for the project.
    """

    def save_model(self, path: str | Path) -> None:
        """Persist the UMAP instance to disk in binary form.

        Parameters
        ----------
        path : str | Path
            Filesystem path where the serialized model will be written.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(path: str | Path) -> "StyleUMAP":
        """Load and return a serialized StyleUMAP instance from disk.

        This static helper reads a previously serialized StyleUMAP (or
        compatible UMAP) object from the specified filesystem path and
        returns the deserialized instance. Implemented as a staticmethod to
        allow loading without requiring an existing instance.

        Parameters
        ----------
        path : str | Path
            Filesystem path from which to read the serialized model.

        Returns
        -------
        StyleUMAP
            The deserialized StyleUMAP instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)


def position_to_vector(fen: str, move: str) -> torch.Tensor:
    """Convert a FEN string and a UCI move into a flattened tensor representation.

    The function constructs the chess.Board from the provided FEN, converts the
    board state immediately before and after the given move into tensor form
    via `board_to_tensor`, concatenates these tensors along the channel dimension,
    and returns a flattened 1-D tensor suitable for model input.

    Parameters
    ----------
    fen : str
        Forsyth–Edwards Notation string describing the board state.
    move : str
        Move in UCI format to apply to the board.

    Returns
    -------
    torch.Tensor
        A one-dimensional tensor representing the concatenation of the board
        encoding before and after the provided move.
    """
    board = chess.Board(fen)
    board_before = board_to_tensor(board)
    board.push_uci(move)
    board_after = board_to_tensor(board)

    vector = torch.cat((board_before, board_after), dim=0).flatten()

    return vector
