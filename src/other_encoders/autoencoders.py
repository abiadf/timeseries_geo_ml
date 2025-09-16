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
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import Losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


# class TrainAutoencoder:
#     """Class dealing with training the autoencoder, measured by the loss"""
#     def _train_epoch(self, device: torch.device, autoencoder: BaseAutoencoder, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
#         autoencoder.train() # set to train mode
#         epoch_loss = 0
#         for data in train_loader:
#             x_input, _ = data
#             x_input    = x_input.to(device)
#             optimizer.zero_grad()
#             x_reconstructed = autoencoder(x_input, mode="reconstruct")
#             loss            = Losses.compute_MSE_loss(x_input, x_reconstructed)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         return epoch_loss / len(train_loader)

#     def _validation_epoch(self, device: torch.device, autoencoder: BaseAutoencoder, validation_loader: DataLoader) -> float:
#         autoencoder.eval()
#         val_loss_total = 0
#         with torch.no_grad(): # disables gradient tracking
#             for data in validation_loader:
#                 x_input, _      = data
#                 x_input         = x_input.to(device)
#                 x_reconstructed = autoencoder(x_input, mode="reconstruct")
#                 validation_loss = Losses.compute_MSE_loss(x_input, x_reconstructed)
#                 val_loss_total += validation_loss.item()
#         return val_loss_total / len(validation_loader)

#     def train_autoencoder(self, device: torch.device, autoencoder: BaseAutoencoder, epochs: int, train_loader: DataLoader,
#                           optimizer: optim.Optimizer, scheduler, validation_loader: DataLoader = None, patience: int = 15) -> float:
#         """Trains the autoencoder for a specified # of epochs, with optional early stopping. A separate validation dataset, not used during training, is used to evaluate the model’s performance during training, helping to monitor the model’s ability to generalize and avoid overfitting. Args:
#             - autoencoder (Autoencoder): An instance of the Autoencoder class to train.
#             - epochs (int): Number of training epochs
#             - x_input (torch.Tensor): The input data tensor
#             - optimizer (torch.optim.Optimizer): The optimizer to use for training
#             - validation_input (torch.Tensor): Optional validation data (same shape as x_input)
#             - patience (int): Early stopping patience in epochs"""
#         autoencoder.to(device)
#         best_loss         = float('inf')
#         epochs_no_improve = 0

#         for epoch in range(epochs):
#             avg_epoch_loss = self._train_epoch(device, autoencoder, train_loader, optimizer)
#             if (epoch + 1) % 20 == 0:
#                 print(f'Epoch [{epoch+1}/{epochs}], training loss: {avg_epoch_loss:.4f}')

#             # Early stopping logic (if validation_loader is provided)
#             if validation_loader is not None:
#                 avg_val_loss = self._validation_epoch(device, autoencoder, validation_loader)
#                 if (epoch + 1) % 20 == 0:
#                     print(f'Validation loss: {avg_val_loss:.4f}')
#                 scheduler.step(avg_val_loss)
#                 if avg_val_loss < best_loss:
#                     best_loss        = avg_val_loss
#                     epochs_no_improve= 0
#                 else:
#                     epochs_no_improve += 1
#                 if epochs_no_improve  >= patience:
#                     print("Early stopping triggered")
#                     break
#             else:
#                 scheduler.step(avg_epoch_loss)
#                 if avg_epoch_loss < best_loss:
#                     best_loss = avg_epoch_loss
#         autoencoder.eval()
#         return best_loss


# class TrainVAE:
#     """Trainer class for Variational Autoencoder (VAE)."""
#     def _train_epoch(self, device, vae, train_loader, optimizer):
#         vae.train()
#         epoch_loss = 0
#         for x_input, _ in train_loader:
#             x_input = x_input.to(device)
#             optimizer.zero_grad()
            
#             x_recon, mu, logvar = vae(x_input)
#             # recon_loss  = nn.functional.mse_loss(x_recon, x_input, reduction='sum')
#             recon_loss = nn.functional.mse_loss(x_recon, x_input, reduction='mean') * x_input.size(0)
#             kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#             loss        = recon_loss + kl_div_loss
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         return epoch_loss / len(train_loader.dataset)
    
#     def _validation_epoch(self, device, vae, val_loader):
#         vae.eval()
#         val_loss_total = 0
#         with torch.no_grad():
#             for x_input, _ in val_loader:
#                 x_input     = x_input.to(device)
#                 x_recon, mu, logvar = vae(x_input)
#                 # recon_loss = nn.functional.mse_loss(x_recon, x_input, reduction='sum')
#                 recon_loss  = nn.functional.mse_loss(x_recon, x_input, reduction='mean') * x_input.size(0)
#                 kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#                 val_loss_total += (recon_loss + kl_div_loss).item()
#         return val_loss_total / len(val_loader.dataset)
    
#     def train_vae(self, device, vae, epochs, train_loader, optimizer, scheduler=None, val_loader=None, patience=15, save_path="best_vae.pth"):
#         best_loss         = float('inf')
#         epochs_no_improve = 0
#         best_state_dict   = None
        
#         for epoch in range(epochs):
#             avg_loss = self._train_epoch(device, vae, train_loader, optimizer)
#             if (epoch + 1) % 20 == 0:
#                 print(f'Epoch [{epoch+1}/{epochs}] train loss: {avg_loss:.4f}')
            
#             if val_loader is not None:
#                 val_loss = self._validation_epoch(device, vae, val_loader)
#                 if (epoch + 1) % 20 == 0:
#                     print(f'Validation loss: {val_loss:.4f}')
#                 if scheduler:
#                     scheduler.step(val_loss)                    
#                 if val_loss < best_loss:
#                     best_loss         = val_loss
#                     epochs_no_improve = 0
#                     best_state_dict   = vae.state_dict()
#                     torch.save(best_state_dict, save_path)
#                     if (epoch + 1) % 20 == 0:
#                         print(f"Saved best model (val_loss={val_loss:.4f})")
#                 else:
#                     epochs_no_improve += 1
                    
#                 if epochs_no_improve >= patience:
#                     print("⏹ Early stopping triggered")
#                     break
#             elif scheduler:
#                 scheduler.step(avg_loss)
#                 if avg_loss < best_loss:
#                     best_loss       = avg_loss
#                     best_state_dict = vae.state_dict()
#                     torch.save(best_state_dict, save_path)
        
#         # restore best model at the end
#         if best_state_dict:
#             vae.load_state_dict(best_state_dict)
#         vae.eval()
#         return best_loss


# class TrainConditionalVAE:
#     """Trainer class for Conditional Variational Autoencoder (CVAE)"""
#     def _train_epoch(self, device, cvae, train_loader, optimizer):
#         cvae.train()
#         epoch_loss = 0
#         for batch in train_loader:
#             if len(batch) == 2:
#                 y, x = batch
#             else:
#                 y = batch[0]
#                 x = None
#             y = y.to(device)
#             x = x.to(device) if x is not None else None

#             optimizer.zero_grad()
#             y_hat, mu, logvar = cvae(y, x)
            
#             # reconstruction + KL
#             recon_loss  = nn.functional.mse_loss(y_hat, y, reduction='mean') * y.size(0)
#             kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#             loss        = recon_loss + kl_div_loss
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         return epoch_loss / len(train_loader.dataset)
    
#     def _validation_epoch(self, device, cvae, val_loader):
#         cvae.eval()
#         val_loss_total = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 if len(batch) == 2:
#                     y, x = batch
#                 else:
#                     y = batch[0]
#                     x = None
#                 y = y.to(device)
#                 x = x.to(device) if x is not None else None

#                 y_hat, mu, logvar= cvae(y, x)
#                 recon_loss       = nn.functional.mse_loss(y_hat, y, reduction='mean') * y.size(0)
#                 kl_div_loss      = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#                 val_loss_total  += (recon_loss + kl_div_loss).item()
#         return val_loss_total / len(val_loader.dataset)
    
#     def train_cvae(self, device, cvae, epochs, train_loader, optimizer, scheduler=None, val_loader=None, patience=15, save_path="best_cvae.pth"):
#         best_loss         = float('inf')
#         epochs_no_improve = 0
#         best_state_dict   = None

#         for epoch in range(epochs):
#             avg_loss = self._train_epoch(device, cvae, train_loader, optimizer)
#             if (epoch + 1) % 20 == 0:
#                 print(f'Epoch [{epoch+1}/{epochs}] train loss: {avg_loss:.4f}')
#             if val_loader is not None:
#                 val_loss = self._validation_epoch(device, cvae, val_loader)
#                 if (epoch + 1) % 20 == 0:
#                     print(f'Validation loss: {val_loss:.4f}')
#                 if scheduler:
#                     scheduler.step(val_loss)
#                 if val_loss < best_loss:
#                     best_loss        = val_loss
#                     epochs_no_improve= 0
#                     best_state_dict  = cvae.state_dict()
#                     torch.save(best_state_dict, save_path)
#                     if (epoch + 1) % 20 == 0:
#                         print(f"Saved best model (val_loss={val_loss:.4f})")
#                 else:
#                     epochs_no_improve += 1
#                 if epochs_no_improve >= patience:
#                     print("⏹ Early stopping triggered")
#                     break
#             elif scheduler:
#                 scheduler.step(avg_loss)
#                 if avg_loss < best_loss:
#                     best_loss       = avg_loss
#                     best_state_dict = cvae.state_dict()
#                     torch.save(best_state_dict, save_path)
#         if best_state_dict:
#             cvae.load_state_dict(best_state_dict)
#         cvae.eval()
#         return best_loss


# TODO to improve AE design
# 	1.	Layer size / depth: Increase/decrease # of layers and their sizes
# 	2.	Activation function: Replace ReLU with LeakyReLU/PReLU/SELU/GELU/Tanh/Sigmoid...
# 	3.	Skip connections/residual connections, ONLY if network >5 layers
#   4. change optimizer from Adam to AdamW/RMSprop/SGD+momentum/LAMB/Adabelief/Lion
#  (usually AdamW is best)
#   5. use regulariation techniques
#   6. use learning rate scheduler