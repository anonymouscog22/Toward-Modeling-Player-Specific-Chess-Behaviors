"""Per-player style embedding helper for integration with the Maia backbone.

This module provides `PlayerStyleEmbedding`, a thin wrapper that composes an
existing Elo-based embedding table (provided by the Maia model) with a small,
learnable per-player embedding matrix. The combined embedding space enables the
Maia model to represent both canonical Elo categories and repository-specific
player identities within a single embedding tensor.
"""

from typing import Any

import torch
import torch.nn as nn


class PlayerStyleEmbedding(nn.Embedding):
    """Compose Maia's Elo embeddings with trainable per-player embeddings.

    Parameters
    ----------
    elo_embeddings : nn.Embedding
        Pre-existing embedding module from the Maia backbone (indexed by Elo
        category).
    n_players : int
        Number of project-specific players to allocate additional embeddings for.

    Behavior
    --------
    The module constructs an embedding space of size (num_maia_embeddings +
    n_players). When called with an input tensor of indices, indices <=
    `max_maia_idx` are mapped to the fixed Maia Elo embeddings, whereas indices
    > `max_maia_idx` are mapped to the corresponding learnable per-player
    embeddings.
    """

    def __init__(self, elo_embeddings: nn.Embedding, n_players: int) -> None:
        total_embeddings = elo_embeddings.num_embeddings + n_players
        super().__init__(
            num_embeddings=total_embeddings, embedding_dim=elo_embeddings.embedding_dim
        )

        # Prevent accidental reuse of the parent `Embedding` storage while maintaining
        # compatibility with modules that introspect `weight`.
        self.weight = nn.Parameter(torch.empty(0))

        self.elo_embeddings: nn.Embedding = elo_embeddings
        self.elo_embeddings.requires_grad_(False)

        self.max_maia_idx: int = elo_embeddings.num_embeddings - 1
        self.dim: int = elo_embeddings.embedding_dim

        self.players_embeddings = nn.Embedding(n_players, self.dim)

        # Initialize project embeddings from the most representative Elo category
        with torch.no_grad():
            best_weights: Any = (
                self.elo_embeddings.weight[self.max_maia_idx].detach().clone()
            )
            self.players_embeddings.weight.data = best_weights.repeat(n_players, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return embeddings for the provided index tensor.

        Parameters
        ----------
        input : torch.Tensor
            Tensor of integer indices. Indices <= `max_maia_idx` are interpreted
            as Maia Elo categories; indices > `max_maia_idx` are interpreted as
            project-specific player indices (offset by `max_maia_idx + 1`).

        Returns
        -------
        torch.Tensor
            A tensor of shape (*input.shape, embedding_dim) containing the
            corresponding embedding vectors.
        """
        is_player = input > self.max_maia_idx

        out = torch.zeros(*input.shape, self.dim, device=input.device)

        if (~is_player).any():
            out[~is_player] = self.elo_embeddings(input[~is_player])

        if is_player.any():
            shifted_indices = input[is_player] - (self.max_maia_idx + 1)
            out[is_player] = self.players_embeddings(shifted_indices)

        return out
