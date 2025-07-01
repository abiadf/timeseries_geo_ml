"""Module designing the autoencoder (AE) of `undercomplete` type, consisting of 2 components, encoder and decoder, both feedforward NN. The encoder compresses input data into latent space, and decoder aims to reconstruct the input data from latent space. Because the latent data is compressed, the encoder's output dimensions are less than its input dimensions (the opposite holds for the decoder). The AE's goodness is measured with a 'reconstruction loss' between input and output data; we iterate until the error is low enough.
Useful hyperparams:
1. # layers for encoder/decoder NN
2. # nodes for each layer
3. size of latent space (smaller = more info lost)"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import Losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Autoencoder(nn.Module):
    """Autoencoder class containing encoder, decoder, and forward pass
    Attributes:
        - input_size (int): Dimensionality of the input data
        - layer1_dim (int): Size of the first hidden layer in encoder/decoder
        - layer2_dim (int): Size of the second hidden layer in encoder/decoder
        - latent_dim (int): Size of the latent representation
        - dropout_prob (float): dropout probability
        - encoder (nn.Sequential): encoder network
        - decoder (nn.Sequential): decoder network"""
    
    def __init__(self, input_size: int, layer1_dim: int, layer2_dim: int,
                 latent_dim: int, dropout_prob: float):
        """Initializes Autoencoder"""

        super(Autoencoder, self).__init__()
        self.input_size   = input_size
        self.layer1_dim   = layer1_dim
        self.layer2_dim   = layer2_dim
        self.latent_dim   = latent_dim
        self.dropout_prob = dropout_prob

        self.encoder = self._build_encoder().to(device)
        self.decoder = self._build_decoder().to(device)

    def _build_encoder(self) -> nn.Sequential:
        """Builds the encoder network as torch sequential model. Currently 2 layers + batchnorm and dropout
        - input_size: # dataset features
        - nn.Sequential: encoder network"""
        
        encoder = nn.Sequential(
                    nn.Linear(self.input_size, self.layer1_dim),
                    nn.BatchNorm1d(self.layer1_dim),
                    # nn.ReLU(),
                    nn.LeakyReLU(negative_slope=0.05), # LeakyReLU instead of ReLU
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(self.layer1_dim, self.layer2_dim),
                    nn.BatchNorm1d(self.layer2_dim),
                    # nn.ReLU(),
                    nn.LeakyReLU(negative_slope=0.05), # LeakyReLU instead of ReLU
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(self.layer2_dim, self.latent_dim))
        return encoder

    def _build_decoder(self) -> nn.Sequential:
        """Builds decoder network as torch sequential model
        - nn.Sequential: decoder network"""

        decoder = nn.Sequential(
                    nn.Linear(self.latent_dim, self.layer2_dim),
                    nn.BatchNorm1d(self.layer2_dim),
                    nn.LeakyReLU(negative_slope=0.05), # LeakyReLU instead of ReLU
                    # nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(self.layer2_dim, self.layer1_dim),
                    nn.BatchNorm1d(self.layer1_dim),
                    nn.LeakyReLU(negative_slope=0.05), # LeakyReLU instead of ReLU
                    # nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(self.layer1_dim, self.input_size))
        return decoder

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder (encoder + decoder).
        NOTE we also add a skip connection
        - x_input (torch.Tensor): input tensor to encode and reconstruct
        - encoder (nn.Module): encoder model
        - decoder (nn.Module): decoder model
        - output (torch.Tensor): Reconstructed output"""
        x_input         = x_input.to(device)
        z_latent        = self.encoder(x_input)
        x_reconstructed = self.decoder(z_latent)

        # skip connection
        # x_reconstructed = x_reconstructed + x_input
        return x_reconstructed

    # ====================
    # LDM portion
    def encode_to_latent(self, x_input: torch.Tensor) -> torch.Tensor:
        """[for LDM] Encodes the input into the latent space"""
        x_input  = x_input.to(device)
        z_latent = self.encoder(x_input)
        return z_latent

    def decode_from_latent(self, z_latent: torch.Tensor) -> torch.Tensor:
        """[for LDM] Decodes the latent representation back to the data space"""
        z_latent        = z_latent.to(device)
        x_reconstructed = self.decoder(z_latent)
        return x_reconstructed


class TrainAutoencoder:
    """Class dealing with training the autoencoder, measured by the loss"""
    def _train_epoch(self, device: torch.device, autoencoder: Autoencoder, train_loader: DataLoader, optimizer: optim.Optimizer) -> float:
        autoencoder.train() # set to train mode
        epoch_loss = 0
        for data in train_loader:
            x_input, _ = data
            x_input    = x_input.to(device)
            optimizer.zero_grad()
            x_reconstructed = autoencoder(x_input)
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
                x_reconstructed = autoencoder(x_input)
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
            if (epoch+1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], training loss: {avg_epoch_loss:.4f}')

            # Early stopping logic (if validation_loader is provided)
            if validation_loader is not None:
                avg_val_loss = self._validation_epoch(device, autoencoder, validation_loader)
                if (epoch+1) % 25 == 0:
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