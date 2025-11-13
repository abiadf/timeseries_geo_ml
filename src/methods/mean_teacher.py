"""
Mean Teacher Implementation for Semi-Supervised Learning

This module implements the Mean Teacher algorithm for semi-supervised learning
with tabular data using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from typing import Tuple, Optional, Union


class TabularRegressor(nn.Module):
    """
    Multi-layer perceptron for tabular regression tasks.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output targets
        hidden_dims: List of hidden layer dimensions
        dropout_ratio: Dropout probability
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [64, 32], dropout_ratio: float = 0.15):
        super().__init__()
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_ratio))
            prev = h
            
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanTeacher:
    """
    Mean Teacher implementation for semi-supervised learning.
    
    The Mean Teacher approach uses an exponential moving average (EMA) teacher model
    to provide consistent targets for unlabeled data during training.
    """
    
    def __init__(
        self,
        batch_size: int = 16,
        dropout_ratio: float = 0.15,
        ema_decay: float = 0.9,
        hidden_dims: list = [64, 32],
        lambda_u: float = 0.18,
        learning_rate: float = 1e-3,
        num_epochs: int = 60,
        student_noise: float = 0.05,
        adam_weight_decay: float = 1e-4,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Mean Teacher trainer.
        
        Args:
            batch_size: Batch size for training
            dropout_ratio: Dropout probability for the model
            ema_decay: EMA decay factor for teacher model update
            hidden_dims: Hidden layer dimensions for the neural network
            lambda_u: Weight for consistency (unsupervised) loss
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
            student_noise: Noise level added to student inputs
            adam_weight_decay: Weight decay for AdamW optimizer
            device: PyTorch device (cuda/cpu)
        """
        self.batch_size = batch_size
        self.dropout_ratio = dropout_ratio
        self.ema_decay = ema_decay
        self.hidden_dims = hidden_dims
        self.lambda_u = lambda_u
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.student_noise = student_noise
        self.adam_weight_decay = adam_weight_decay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.student = None
        self.teacher = None
        self.optimizer = None
        self.scaler = None

    def prepare_data(self, X_labeled: Union[np.ndarray, pd.DataFrame], y_labeled: Union[np.ndarray, pd.DataFrame], X_unlabeled: Union[np.ndarray, pd.DataFrame]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare and scale the data for training.
        
        Args:
            X_labeled: Labeled feature data
            y_labeled: Labeled target data
            X_unlabeled: Unlabeled feature data
            
        Returns:
            Tuple of (X_labeled_tensor, y_labeled_tensor, X_unlabeled_tensor)
        """
        # Handle pandas DataFrames
        if isinstance(X_labeled, pd.DataFrame):
            # Remove NaN columns
            X_labeled = X_labeled.dropna(axis=1, how='all')
            # Remove constant columns
            X_labeled = X_labeled.loc[:, X_labeled.nunique() > 1]
            # Fill NaN values with mean
            X_labeled = X_labeled.fillna(X_labeled.mean())
            
        if isinstance(X_unlabeled, pd.DataFrame):
            # Remove NaN columns
            X_unlabeled = X_unlabeled.dropna(axis=1, how='all')
            # Remove constant columns
            X_unlabeled = X_unlabeled.loc[:, X_unlabeled.nunique() > 1]
            # Fill NaN values with mean
            X_unlabeled = X_unlabeled.fillna(X_unlabeled.mean())
            
        # Convert to numpy if needed
        if hasattr(X_labeled, 'values'):
            X_labeled = X_labeled.values
        if hasattr(X_unlabeled, 'values'):
            X_unlabeled = X_unlabeled.values
        if hasattr(y_labeled, 'values'):
            y_labeled = y_labeled.values
            
        # Handle numpy arrays - basic cleaning
        if isinstance(X_labeled, np.ndarray):
            # Remove NaN rows if present
            if np.isnan(X_labeled).any():
                # Fill with column means
                col_means = np.nanmean(X_labeled, axis=0)
                for i in range(X_labeled.shape[1]):
                    X_labeled[np.isnan(X_labeled[:, i]), i] = col_means[i]
                    
        if isinstance(X_unlabeled, np.ndarray):
            if np.isnan(X_unlabeled).any():
                col_means = np.nanmean(X_unlabeled, axis=0)
                for i in range(X_unlabeled.shape[1]):
                    X_unlabeled[np.isnan(X_unlabeled[:, i]), i] = col_means[i]
            
        # Scale features
        self.scaler = StandardScaler()
        all_features = np.vstack([X_labeled, X_unlabeled])
        self.scaler.fit(all_features)
        
        X_labeled_scaled = self.scaler.transform(X_labeled)
        X_unlabeled_scaled = self.scaler.transform(X_unlabeled)
        
        # Convert to tensors
        X_labeled_tensor = torch.tensor(X_labeled_scaled, dtype=torch.float32).to(self.device)
        y_labeled_tensor = torch.tensor(y_labeled, dtype=torch.float32).to(self.device)
        X_unlabeled_tensor = torch.tensor(X_unlabeled_scaled, dtype=torch.float32).to(self.device)
        
        # Validate no NaN or inf values
        assert not torch.isnan(X_labeled_tensor).any(), "NaN values in labeled features"
        assert not torch.isnan(X_unlabeled_tensor).any(), "NaN values in unlabeled features"
        assert not torch.isnan(y_labeled_tensor).any(), "NaN values in labeled targets"
        assert not torch.isinf(X_labeled_tensor).any(), "Inf values in labeled features"
        assert not torch.isinf(X_unlabeled_tensor).any(), "Inf values in unlabeled features"
        assert not torch.isinf(y_labeled_tensor).any(), "Inf values in labeled targets"
        
        return X_labeled_tensor, y_labeled_tensor, X_unlabeled_tensor

    def initialize_models(self, input_dim: int, output_dim: int):
        """
        Initialize student and teacher models.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
        """
        self.student = TabularRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            dropout_ratio=self.dropout_ratio
        ).to(self.device)
        
        self.teacher = TabularRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            dropout_ratio=self.dropout_ratio
        ).to(self.device)
        
        # Initialize teacher with student weights
        self.teacher.load_state_dict(self.student.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=self.adam_weight_decay
        )

    def update_teacher_weights(self):
        """Update teacher model weights using exponential moving average."""
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
                teacher_param.data.mul_(self.ema_decay).add_(student_param.data * (1 - self.ema_decay))

    def train_epoch(self, labeled_loader: DataLoader, unlabeled_loader: DataLoader, epoch: int) -> float:
        """
        Train for one epoch using Mean Teacher algorithm.
        
        Args:
            labeled_loader: DataLoader for labeled data
            unlabeled_loader: DataLoader for unlabeled data
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.student.train()
        self.teacher.eval()
        
        # Gradually ramp up consistency loss weight
        lambda_u_epoch = self.lambda_u * min(1.0, epoch / 5)
        
        total_loss = 0
        unlabeled_iter = iter(unlabeled_loader)
        
        for labeled_x, labeled_y in labeled_loader:
            try:
                (unlabeled_x,) = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (unlabeled_x,) = next(unlabeled_iter)

            # Supervised loss on labeled data
            pred_labeled = self.student(labeled_x)
            loss_supervised = F.mse_loss(pred_labeled, labeled_y)

            # Consistency loss on unlabeled data
            with torch.no_grad():
                teacher_preds = self.teacher(unlabeled_x)
            
            # Add noise to student input for consistency regularization
            noisy_unlabeled_x = unlabeled_x + self.student_noise * torch.randn_like(unlabeled_x)
            student_preds = self.student(noisy_unlabeled_x)
            loss_consistency = F.mse_loss(student_preds, teacher_preds)

            # Total loss
            total_loss_batch = loss_supervised + lambda_u_epoch * loss_consistency

            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()

            # Update teacher weights with EMA
            self.update_teacher_weights()
            
            total_loss += total_loss_batch.item()

        return total_loss / len(labeled_loader)

    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, X_unlabeled: np.ndarray):
        """
        Train the Mean Teacher model.
        
        Args:
            X_labeled: Labeled feature data
            y_labeled: Labeled target data
            X_unlabeled: Unlabeled feature data
        """
        # Prepare data
        X_labeled_tensor, y_labeled_tensor, X_unlabeled_tensor = self.prepare_data(
            X_labeled, y_labeled, X_unlabeled
        )
        
        # Initialize models
        self.initialize_models(X_labeled_tensor.shape[1], y_labeled_tensor.shape[1])
        
        # Create data loaders
        labeled_loader = DataLoader(
            TensorDataset(X_labeled_tensor, y_labeled_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )
        unlabeled_loader = DataLoader(
            TensorDataset(X_unlabeled_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            avg_loss = self.train_epoch(labeled_loader, unlabeled_loader, epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")

    def evaluate_rmse(self, X: torch.Tensor, y: torch.Tensor, model: Optional[nn.Module] = None) -> float:
        """
        Evaluate RMSE on given data.
        
        Args:
            X: Input features
            y: Target values
            model: Model to evaluate (uses teacher if None)
            
        Returns:
            RMSE score
        """
        if model is None:
            model = self.teacher
            
        model.eval()
        with torch.no_grad():
            preds = model(X).cpu().numpy()
        rmse = root_mean_squared_error(y.cpu().numpy(), preds)
        return rmse

    def predict(self, X: np.ndarray, use_teacher: bool = True) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            use_teacher: Whether to use teacher model (default) or student
            
        Returns:
            Predictions
        """
        if self.scaler is None:
            raise ValueError("Model must be trained before making predictions")
            
        # Prepare input
        if hasattr(X, 'values'):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Choose model
        model = self.teacher if use_teacher else self.student
        model.eval()
        
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
            
        return predictions


def pseudo_labeling(teacher: nn.Module, X_unlabeled: torch.Tensor, threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pseudo-labels for unlabeled data using confidence thresholding.
    
    Args:
        teacher: Trained teacher model
        X_unlabeled: Unlabeled input data
        threshold: Confidence threshold for pseudo-label selection
        
    Returns:
        Tuple of (high_confidence_X, pseudo_labels)
    """
    teacher.eval()
    with torch.no_grad():
        preds = teacher(X_unlabeled)
    
    preds_np = preds.cpu().numpy()
    # Use confidence = inverse of prediction std dev as example
    conf = 1 / (preds_np.std(axis=1) + 1e-6)
    high_conf_mask = conf > threshold
    
    return X_unlabeled[high_conf_mask], preds.detach()[high_conf_mask]


def retrain_on_combined_data(
    X_labeled: torch.Tensor,
    y_labeled: torch.Tensor,
    X_pseudo: torch.Tensor,
    y_pseudo: torch.Tensor,
    input_dim: int,
    output_dim: int,
    hidden_dims: list = [64, 32],
    dropout_ratio: float = 0.15,
    batch_size: int = 32,
    epochs: int = 30,
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None
) -> TabularRegressor:
    """
    Retrain a model on combined labeled and pseudo-labeled data.
    
    Args:
        X_labeled: Labeled features
        y_labeled: Labeled targets
        X_pseudo: Pseudo-labeled features
        y_pseudo: Pseudo-labeled targets
        input_dim: Number of input features
        output_dim: Number of output targets
        hidden_dims: Hidden layer dimensions
        dropout_ratio: Dropout probability
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: PyTorch device
        
    Returns:
        Trained model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Combine labeled and pseudo-labeled data
    combined_X = torch.cat([X_labeled, X_pseudo], dim=0)
    combined_y = torch.cat([y_labeled, y_pseudo], dim=0)
    
    # Create data loader
    combined_loader = DataLoader(
        TensorDataset(combined_X, combined_y),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize new model
    model = TabularRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout_ratio=dropout_ratio
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in combined_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = F.mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Retrain Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(combined_loader):.4f}")
    
    return model
