"""Module designing the autoencoder (AE) of `undercomplete` type, consisting of 2 components, encoder and decoder, both feedforward NN. The encoder compresses input data into latent space, and decoder aims to reconstruct the input data from latent space. Because the latent data is compressed, the encoder's output dimensions are less than its input dimensions (the opposite holds for the decoder). The AE's goodness is measured with a 'reconstruction loss' between input and output data; we iterate until the error is low enough.
Useful hyperparams:
1. # layers for encoder/decoder NN
2. # nodes for each layer
3. size of latent space (smaller = more info lost)"""

from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import Losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Autoencoder(nn.Module):
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
        self.pred_hidden_dim= pred_hidden_dim
        self.use_projection = projection_dim > 0
        self.encoder        = self._build_layers(layer_dims, is_decoder=False)  # full forward pass including latent dim
        self.decoder        = self._build_layers(layer_dims[::-1], is_decoder=True)  # reverse for decoder

        if self.use_projection:
            self.projection_head = nn.Sequential(
                nn.Linear(layer_dims[-1], projection_dim), # latent space Z -> projection space H
                nn.BatchNorm1d(projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim)) #residual transformation
        else:
            self.projection_head = nn.Identity()

        # Prediction head (from latent or projection)
        if pred_dim > 0:
            self.prediction_head = nn.Sequential(
                nn.Linear(layer_dims[-1], pred_hidden_dim), # from latent space Z
                nn.ReLU(),
                nn.Linear(pred_hidden_dim, pred_dim)) # final prediction layer
        else:
            self.prediction_head = None

    def _build_layers(self, dims: list[int], is_decoder: bool = False) -> nn.Sequential:
        """Builds a sequential block (encoder or decoder) from a list of dimensions.
            - dims (list[int]): List of layer sizes
            - is_decoder (bool): Whether building decoder (affects final layer behavior)
            - nn.Sequential: Fully constructed block"""
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = (i == len(dims) - 2)
            if not is_last:
                layers.extend([
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(0.05),
                    nn.Dropout(self.dropout_prob)])
        return nn.Sequential(*layers)

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
        x = x.to(device)
        z = self.encoder(x)
        if mode == "reconstruct":
            return self.decoder(z)
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


class TrainAutoencoder:
    """Class dealing with training the autoencoder, measured by the loss"""
    def _train_epoch(self, device: torch.device, autoencoder: Autoencoder, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        autoencoder.train() # set to train mode
        epoch_loss = 0
        for data in train_loader:
            x_input, _ = data
            x_input    = x_input.to(device)
            optimizer.zero_grad()
            x_reconstructed = autoencoder(x_input, mode="reconstruct")
            loss = Losses.compute_MSE_loss(x_input, x_reconstructed)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def _validation_epoch(self, device: torch.device, autoencoder: Autoencoder, validation_loader: DataLoader) -> float:
        autoencoder.eval()
        val_loss_total = 0
        with torch.no_grad(): # disables gradient tracking
            for data in validation_loader:
                x_input, _      = data
                x_input         = x_input.to(device)
                x_reconstructed = autoencoder(x_input, mode="reconstruct")
                validation_loss = Losses.compute_MSE_loss(x_input, x_reconstructed)
                val_loss_total += validation_loss.item()
        return val_loss_total / len(validation_loader)

    def train_autoencoder(self, device: torch.device, autoencoder: Autoencoder, epochs: int, train_loader: DataLoader,
                          optimizer: optim.Optimizer, scheduler, validation_loader: DataLoader = None, patience: int = 15) -> float:
        """Trains the autoencoder for a specified # of epochs, with optional early stopping. A separate validation dataset, not used during training, is used to evaluate the model’s performance during training, helping to monitor the model’s ability to generalize and avoid overfitting. Args:
            - autoencoder (Autoencoder): An instance of the Autoencoder class to train.
            - epochs (int): Number of training epochs
            - x_input (torch.Tensor): The input data tensor
            - optimizer (torch.optim.Optimizer): The optimizer to use for training
            - validation_input (torch.Tensor): Optional validation data (same shape as x_input)
            - patience (int): Early stopping patience in epochs"""

        autoencoder.to(device)
        best_loss         = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            avg_epoch_loss = self._train_epoch(device, autoencoder, train_loader, optimizer)
            if (epoch+1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], training loss: {avg_epoch_loss:.4f}')

            # Early stopping logic (if validation_loader is provided)
            if validation_loader is not None:
                avg_val_loss = self._validation_epoch(device, autoencoder, validation_loader)
                if (epoch+1) % 20 == 0:
                    print(f'Validation loss: {avg_val_loss:.4f}')
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_loss:
                    best_loss        = avg_val_loss
                    epochs_no_improve= 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve  >= patience:
                    print("Early stopping triggered")
                    break
            else:
                scheduler.step(avg_epoch_loss)
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
        autoencoder.eval()
        return best_loss

# TODO to improve AE design
# 	1.	Layer size / depth: Increase/decrease # of layers and their sizes
# 	2.	Activation function: Replace ReLU with LeakyReLU/PReLU/SELU/GELU/Tanh/Sigmoid...
# 	3.	Skip connections/residual connections, ONLY if network >5 layers
#   4. change optimizer from Adam to AdamW/RMSprop/SGD+momentum/LAMB/Adabelief/Lion
#  (usually AdamW is best)
#   5. use regulariation techniques
#   6. use learning rate scheduler