"""Module designing the autoencoder (AE) of `undercomplete` type, consisting of 2 components, encoder and decoder, both feedforward
NN. The encoder compresses input data into latent space, and decoder aims to reconstruct the input data from latent space. Because
the latent data is compressed, the encoder's output dimensions are less than its input dimensions (the opposite holds for the 
decoder). The AE's goodness is measured with a 'reconstruction loss' between input and output data; we iterate until the error is low enough.
Useful hyperparams:
1. # layers for encoder/decoder NN
2. # nodes for each layer
3. size of latent space (smaller = more info lost)"""

from typing import Union, Tuple

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseAutoencoder(nn.Module):
    """Abstract Autoencoder base class"""
    def __init__(self, negative_slope: float = 0.05):
        super().__init__()
        self.negative_slope = negative_slope

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mode: str = "reconstruct") -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(z)


class SimpleAutoencoder(BaseAutoencoder):
    """Simple 2-layer AE with encoder and decoder modules"""
    def __init__(self, input_size: int, layer1_dim: int, layer2_dim: int,
                 latent_dim: int, dropout_prob: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, layer1_dim),
            nn.BatchNorm1d(layer1_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(dropout_prob),
            nn.Linear(layer1_dim, layer2_dim),
            nn.BatchNorm1d(layer2_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(dropout_prob),
            nn.Linear(layer2_dim, latent_dim))

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layer2_dim),
            nn.BatchNorm1d(layer2_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(dropout_prob),
            nn.Linear(layer2_dim, layer1_dim),
            nn.BatchNorm1d(layer1_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(dropout_prob),
            nn.Linear(layer1_dim, input_size))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class FlexibleAutoencoder(BaseAutoencoder):
    """Flexible-depth Autoencoder with encoder and decoder modules.
    Args:
        layer_dims (list[int]): List of layer sizes, e.g. [input_dim, ..., latent_dim]
        dropout_prob (float): Dropout probability between layers
    Attributes:
        encoder (nn.Sequential): Encoder network
        decoder (nn.Sequential): Decoder network"""
    def __init__(self, layer_dims: list[int], pred_dim: int, dropout_prob: float = 0.05,
                 projection_dim: int = 16, pred_hidden_dim: int = 32, mode: str = "reconstruct"):
        super().__init__()
        self.layer_dims     = layer_dims
        self.dropout_prob   = dropout_prob
        self.mode           = mode
        self.use_projection = projection_dim > 0

        # encoder/decoder
        self.encoder = self._build_layers(layer_dims, is_decoder=False)
        self.decoder = self._build_layers(layer_dims[::-1], is_decoder=True)

        # projection head
        self.projection_head = (
            nn.Sequential(
                nn.Linear(layer_dims[-1], projection_dim),
                nn.LayerNorm(projection_dim), #nn.BatchNorm1d(projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim))
            if self.use_projection else nn.Identity())

        # [optional] prediction head
        self.prediction_head = (
            nn.Sequential(
                nn.Linear(layer_dims[-1], pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(pred_hidden_dim, pred_dim))
            if pred_dim > 0 else None)

    def _build_layers(self, dims: list[int], is_decoder: bool = False) -> nn.Sequential:
        """Builds a sequential block (encoder or decoder) from a list of dimensions.
            - dims (list[int]): List of layer sizes
            - is_decoder (bool): Whether building decoder (affects final layer behavior)
            - nn.Sequential: Fully constructed block"""
        layers_list = []
        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers_list.append(nn.Linear(in_dim, out_dim))
            is_last = (i == len(dims) - 2)
            if not is_last:
                layers_list.extend([
                    nn.LayerNorm(out_dim), # nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(self.negative_slope),
                    nn.Dropout(self.dropout_prob)])
        return nn.Sequential(*layers_list)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, mode: str = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the autoencoder. Z is latent space, H is projection space (in Z)
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
                mode (str): One of {"reconstruct", "projection", "latent"}
                    - "reconstruct": Returns decoded output X̂
                    - "projection" : Returns projected embedding H (after optional projection head)
                    - "latent"     : Returns both latent vector Z and projection H
                    - "predict"    : Returns prediction based on latent Z
            Returns:
                Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                    - If mode="reconstruct" → X̂
                    - If mode="projection"  → H
                    - If mode="latent"      → (Z, H)
                    - If mode="predict"     → y_pred"""
        if mode is None:
            mode = self.mode
        z = self.encode(x)
        if mode == "reconstruct":
            return self.decode(z)
        elif mode == "projection":
            return self.projection_head(z) if self.use_projection else z
        elif mode == "latent":
            h = self.projection_head(z) if self.use_projection else z
            return z, h
        elif mode == "predict":
            if not self.prediction_head:
                raise ValueError("Prediction head not defined")
            return self.prediction_head(z)
        else:
            raise ValueError(f"Invalid mode: {mode}")


class VAE(BaseAutoencoder):
    def __init__(self, input_size: int, hidden_dim: int, latent_dim: int, dropout_prob: float = 0.05):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.latent_dim   = latent_dim
        self.dropout_prob = dropout_prob

        # Encoder: outputs mu and log_var
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(dropout_prob))
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim) # fc = fully connected (layer)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, input_size))

    def encode(self, x: torch.Tensor):
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor):
        stdev = torch.exp(0.5 * log_var)
        eps   = torch.randn_like(stdev)
        return mu + eps * stdev

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decode(z)
        return x_recon, mu, log_var


class ConditionalVAE(nn.Module):
    """Conditional VAE: predicts y, optionally conditioned on x
    y -> encoder -> μ, log σ² -> reparameterize -> z -> decoder -> y_hat
                                           x (optional) ---^  """
    def __init__(self, y_dim: int, x_dim: int = 0, hidden_dim: int = 64, latent_dim: int = 8, dropout: float = 0.05, negative_slope: float = 0.05):
        super().__init__()
        self.y_dim         = y_dim
        self.x_dim         = x_dim
        self.hidden_dim    = hidden_dim
        self.latent_dim    = latent_dim
        self.dropout       = dropout
        self.negative_slope= negative_slope

        # Encoder: only sees y
        self.encoder = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(dropout))
        
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim) # fc = fully connected (layer)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: sees z + optional x
        decoder_input_dim = latent_dim + x_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(self.negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, y_dim))

    def encode(self, y: torch.Tensor):
        h      = self.encoder(y)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        stdev = torch.exp(0.5 * logvar)
        eps   = torch.randn_like(stdev)
        return mu + eps * stdev

    def decode(self, z: torch.Tensor, x: torch.Tensor = None):
        if self.x_dim > 0 and x is not None:
            z = torch.cat([z, x], dim=1)
        return self.decoder(z)

    def forward(self, y: torch.Tensor, x: torch.Tensor = None):
        mu, logvar = self.encode(y)
        z          = self.reparameterize(mu, logvar)
        y_hat      = self.decode(z, x)
        return y_hat, mu, logvar


class DenoisingAE(BaseAutoencoder):
    """Denoising Autoencoder for timeseries. Adds Gaussian noise to input during training."""
    def __init__(self, input_size: int, hidden_dims: list[int], latent_dim: int,
                 dropout_prob: float = 0.05, noise_std: float = 0.1):
        super().__init__()
        self.noise_std    = noise_std
        self.dropout_prob = dropout_prob

        # Encoder
        dims         = [input_size] + hidden_dims + [latent_dim]
        self.encoder = self._build_layers(dims, is_decoder=False)

        # Decoder
        dims_decoder = [latent_dim] + hidden_dims[::-1] + [input_size]
        self.decoder = self._build_layers(dims_decoder, is_decoder=True)

    def _build_layers(self, dims: list[int], is_decoder: bool = False) -> nn.Sequential:
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = (i == len(dims) - 2)
            if not is_last:
                layers.extend([nn.LayerNorm(out_dim),
                               nn.LeakyReLU(self.negative_slope),
                               nn.Dropout(self.dropout_prob)])
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Forward pass. Adds Gaussian noise if add_noise=True (training)."""
        if self.training and add_noise and self.noise_std > 0:
            noise   = torch.randn_like(x) * self.noise_std
            x_noisy = x + noise
        else:
            x_noisy = x
        z = self.encode(x_noisy)
        return self.decode(z)



# TODO to improve AE design
# 	1.	Layer size / depth: Increase/decrease # of layers and their sizes
# 	2.	Activation function: Replace ReLU with LeakyReLU/PReLU/SELU/GELU/Tanh/Sigmoid...
# 	3.	Skip connections/residual connections, ONLY if network >5 layers
#   4. change optimizer from Adam to AdamW/RMSprop/SGD+momentum/LAMB/Adabelief/Lion
#  (usually AdamW is best)
#   5. use regulariation techniques
#   6. use learning rate scheduler