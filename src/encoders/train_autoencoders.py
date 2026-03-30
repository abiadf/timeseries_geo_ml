"""Module training the AE/VAE"""

from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.metrics_utils import Losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainAutoencoder:
    """Class dealing with training the autoencoder, measured by the loss. NOTE: include X in training"""
    def _train_epoch(self, device: torch.device, autoencoder, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        autoencoder.train() # set to train mode
        epoch_loss = 0
        for data in train_loader:
            x_input, _ = data
            x_input    = x_input.to(device)
            optimizer.zero_grad()
            x_reconstructed = autoencoder(x_input, mode="reconstruct")
            loss            = Losses.compute_MSE_loss(x_input, x_reconstructed)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def _validation_epoch(self, device: torch.device, autoencoder, validation_loader: DataLoader) -> float:
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

    def train_autoencoder(self, device: torch.device, autoencoder, epochs: int, train_loader: DataLoader,
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
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], training loss: {avg_epoch_loss:.4f}')

            # Early stopping logic (if validation_loader is provided)
            if validation_loader is not None:
                avg_val_loss = self._validation_epoch(device, autoencoder, validation_loader)
                if (epoch + 1) % 20 == 0:
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


class TrainVAE:
    """Trainer class for Variational Autoencoder (VAE)."""
    def _train_epoch(self, device, vae, train_loader, optimizer):
        vae.train()
        epoch_loss = 0
        for x_input, _ in train_loader:
            x_input = x_input.to(device)
            optimizer.zero_grad()
            
            x_recon, mu, logvar = vae(x_input)
            # recon_loss  = nn.functional.mse_loss(x_recon, x_input, reduction='sum')
            recon_loss = nn.functional.mse_loss(x_recon, x_input, reduction='mean') * x_input.size(0)
            kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss        = recon_loss + kl_div_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader.dataset)
    
    def _validation_epoch(self, device, vae, val_loader):
        vae.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x_input, _ in val_loader:
                x_input     = x_input.to(device)
                x_recon, mu, logvar = vae(x_input)
                # recon_loss = nn.functional.mse_loss(x_recon, x_input, reduction='sum')
                recon_loss  = nn.functional.mse_loss(x_recon, x_input, reduction='mean') * x_input.size(0)
                kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss_total += (recon_loss + kl_div_loss).item()
        return val_loss_total / len(val_loader.dataset)
    
    def train_vae(self, device, vae, epochs, train_loader, optimizer, scheduler=None, val_loader=None, patience=15, save_path="best_vae.pth"):
        best_loss         = float('inf')
        epochs_no_improve = 0
        best_state_dict   = None
        
        for epoch in range(epochs):
            avg_loss = self._train_epoch(device, vae, train_loader, optimizer)
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] train loss: {avg_loss:.4f}')
            
            if val_loader is not None:
                val_loss = self._validation_epoch(device, vae, val_loader)
                if (epoch + 1) % 20 == 0:
                    print(f'Validation loss: {val_loss:.4f}')
                if scheduler:
                    scheduler.step(val_loss)                    
                if val_loss < best_loss:
                    best_loss         = val_loss
                    epochs_no_improve = 0
                    best_state_dict   = vae.state_dict()
                    torch.save(best_state_dict, save_path)
                    if (epoch + 1) % 20 == 0:
                        print(f"Saved best model (val_loss={val_loss:.4f})")
                else:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    print("⏹ Early stopping triggered")
                    break
            elif scheduler:
                scheduler.step(avg_loss)
                if avg_loss < best_loss:
                    best_loss       = avg_loss
                    best_state_dict = vae.state_dict()
                    torch.save(best_state_dict, save_path)
        
        # restore best model at the end
        if best_state_dict:
            vae.load_state_dict(best_state_dict)
        vae.eval()
        return best_loss


class TrainConditionalVAE:
    """Trainer class for Conditional Variational Autoencoder (CVAE). NOTE: always train with X, infer with or without X.
    Note: this class accommodates β-VAE, where β is an optional hyperparam to control the weight of the KL divergence term in the loss function."""
    def _train_epoch(self, device, cvae, train_loader, optimizer, use_beta: bool = False, beta: float = 2.0):
        cvae.train()
        epoch_loss = 0
        for batch in train_loader:
            if len(batch) == 2:
                y, x = batch
            else:
                y = batch[0]
                x = None
            y = y.to(device)
            x = x.to(device) if x is not None else None

            optimizer.zero_grad()
            y_hat, mu, logvar = cvae(y, x)
            
            # reconstruction + KL
            recon_loss  = nn.functional.mse_loss(y_hat, y, reduction='mean') * y.size(0)
            kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss        = recon_loss + (beta * kl_div_loss if use_beta else kl_div_loss) # to make it beta-VAE
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader.dataset)
    
    def _validation_epoch(self, device, cvae, val_loader):
        cvae.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    y, x = batch
                else:
                    y = batch[0]
                    x = None
                y = y.to(device)
                x = x.to(device) if x is not None else None

                y_hat, mu, logvar= cvae(y, x)
                recon_loss       = nn.functional.mse_loss(y_hat, y, reduction='mean') * y.size(0)
                kl_div_loss      = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss_total  += (recon_loss + kl_div_loss).item()
        return val_loss_total / len(val_loader.dataset)
    
    def train_cvae(self, device, cvae, epochs, train_loader, optimizer, scheduler=None, val_loader=None, patience=15, save_path="best_cvae.pth"):
        best_loss         = float('inf')
        epochs_no_improve = 0
        best_state_dict   = None

        for epoch in range(epochs):
            avg_loss = self._train_epoch(device, cvae, train_loader, optimizer)
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}] train loss: {avg_loss:.4f}')
            if val_loader is not None:
                val_loss = self._validation_epoch(device, cvae, val_loader)
                if (epoch + 1) % 20 == 0:
                    print(f'Validation loss: {val_loss:.4f}')
                if scheduler:
                    scheduler.step(val_loss)
                if val_loss < best_loss:
                    best_loss        = val_loss
                    epochs_no_improve= 0
                    best_state_dict  = cvae.state_dict()
                    torch.save(best_state_dict, save_path)
                    if (epoch + 1) % 20 == 0:
                        print(f"Saved best model (val_loss={val_loss:.4f})")
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("⏹ Early stopping triggered")
                    break
            elif scheduler:
                scheduler.step(avg_loss)
                if avg_loss < best_loss:
                    best_loss       = avg_loss
                    best_state_dict = cvae.state_dict()
                    torch.save(best_state_dict, save_path)
        if best_state_dict:
            cvae.load_state_dict(best_state_dict)
        cvae.eval()
        return best_loss

