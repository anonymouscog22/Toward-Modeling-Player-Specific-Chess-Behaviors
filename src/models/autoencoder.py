"""Autoencoder neural network for representation learning and reconstruction.

This module implements a feed-forward autoencoder designed to compress
fixed-length input vectors into a lower-dimensional latent representation and
to reconstruct inputs from that latent code. The architecture is intentionally
simple to facilitate experimentation with latent dimensionality, training
objectives, and downstream analyses.
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Feed-forward autoencoder with an encoder and a symmetric decoder.

    The encoder progressively reduces dimensionality from `input_dim` down to
    `latent_dim`. The decoder mirrors the encoder to reconstruct the original
    input dimensionality.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vectors.
    latent_dim : int, optional
        Dimensionality of the latent representation (default: 128).
    """

    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the reconstruction for input tensor `x`.

        The method encodes `x` to the latent space and decodes it back to the
        original dimensionality, returning the reconstructed tensor.
        """
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation produced by the encoder for `x`."""
        return self.encoder(x)
