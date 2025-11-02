"""All predictions go here"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head: maps latent z to projected space H for contrastive learning
    Notes:
    - Last layer is Linear only, **no BatchNorm** (kills contrastive loss), which is important for NT-Xent / cosine similarity loss.
    - Outputs are normalized with F.normalize to unit vectors for contrastive similarity."""
    def __init__(self, input_dim: int, proj_dim: int, hidden_sizes: list[int] = [256], dropout: float = 0.0):
        super().__init__()
        layers   = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        # Final projection layer: NO BatchNorm here!
        layers.append(nn.Linear(prev_dim, proj_dim))  # final projection
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent z into normalized space H for contrastive loss.
        - z: (batch, latent_dim)
        - h: (batch, proj_dim), L2-normalized"""
        h = self.net(z)
        h = F.normalize(h, dim=1)  # normalize to unit vectors; ensures cosine similarity is meaningful
        return h


class Decoder(nn.Module):
    """Optional decoder to reconstruct the original input from latent embeddings z"""
    def __init__(self, latent_dim: int, output_shape: tuple[int, int], hidden_sizes=[128, 128], dropout: float = 0.0):
        """Args:
            latent_dim: Dimensionality of input latent z
            output_shape: Tuple (time_steps, channels) for reconstruction
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability in hidden layers"""
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_shape[0] * output_shape[1]))  # flatten output
        self.net = nn.Sequential(*layers)
        self.output_shape = output_shape

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass: map latent z → reconstructed X
        Args:
            z: (batch, latent_dim)
        Returns:
            X_hat: (batch, time_steps, channels)"""
        x_hat = self.net(z)
        return x_hat.view(-1, *self.output_shape)  # reshape to (batch, time, channels)


class TorchWrapper(nn.Module):
    """Wraps a non-nn.Module model for PyTorch pipelines.
    Exposes the internal network as `.net` and also `.model` for compatibility."""
    def __init__(self, ts_model):
        super().__init__()
        self.net = ts_model.net if hasattr(ts_model, "net") else ts_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def model(self):
        return self.net
    
