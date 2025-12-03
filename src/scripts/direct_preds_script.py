"Direct pred. (X > y) on dataset with missing labels"
# Train on X_L, predict on X_test (no mention of X_U)

import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Dict, List, Literal, Tuple, Optional
import logging

import numexpr as ne # makes numpy operations faster
import category_encoders as ce
import numpy as np

from src.methods.mlp_heads import make_MLP_regression_head, evaluate_MLP_regressor
from src.utils.metrics_utils import Preds

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")


def eval_mean_row(X_L: np.ndarray, X_test: np.ndarray, y_L: np.ndarray, y_test: np.ndarray,
                  cfg, device: str) -> Tuple[List[float], object]:
    """Train models on row-wise mean features and evaluate (includes MLP)."""
    X_L_2d    = X_L.mean(axis=1).astype(np.float32)
    X_test_2d = X_test.mean(axis=1).astype(np.float32)
    
    # Standard models
    losses, rf_model = Preds().evaluate_models_on_dataset(X_L_2d, y_L, X_test_2d, y_test)
    
    # MLP regression
    embedding_dim = X_L_2d.shape[1]
    dropout = 0
    regression_head = make_MLP_regression_head(embedding_dim,
                                               [cfg.layer1_dim, cfg.layer2_dim, cfg.layer3_dim],
                                               y_L, dropout, device)
    mlp_loss = evaluate_MLP_regressor(regression_head, X_L_2d, X_test_2d, y_L, y_test,
                                      cfg.regressor_epochs, cfg.lr_regressor, device)
    losses.append(mlp_loss)
    return losses, rf_model


def eval_custom_row(X_L: np.ndarray, X_test: np.ndarray, y_L: np.ndarray, y_test: np.ndarray,
                    cfg, device: str, row_number: int = -1) -> Tuple[List[float], object]:
    """Train models on a specific row index per page and evaluate (includes MLP)."""
    X_train_row = X_L[:, row_number, :].astype(np.float32)
    X_test_row  = X_test[:, row_number, :].astype(np.float32)
    
    losses, rf_model = Preds().evaluate_models_on_dataset(X_train_row, y_L, X_test_row, y_test)
    
    # MLP regression
    embedding_dim = X_train_row.shape[1]
    dropout = 0
    regression_head = make_MLP_regression_head(embedding_dim,
                                               [cfg.layer1_dim, cfg.layer2_dim, cfg.layer3_dim],
                                               y_L, dropout, device)
    mlp_loss = evaluate_MLP_regressor(regression_head, X_train_row, X_test_row, y_L, y_test,
                                      cfg.regressor_epochs, cfg.lr_regressor, device)
    losses.append(mlp_loss)
    return losses, rf_model


def eval_random_row(X_L: np.ndarray, X_test: np.ndarray, y_L: np.ndarray, y_test: np.ndarray,
                    cfg, device: str) -> Tuple[List[float], object]:
    """Train models on a random row per page and evaluate (includes MLP)."""
    n_pages, n_rows, _ = X_L.shape
    n_test = X_test.shape[0]
    train_idx = np.random.randint(0, n_rows, size=n_pages)
    test_idx  = np.random.randint(0, n_rows, size=n_test)
    X_train_rand = X_L[np.arange(n_pages), train_idx, :].astype(np.float32)
    X_test_rand  = X_test[np.arange(n_test), test_idx, :].astype(np.float32)
    
    losses, rf_model = Preds().evaluate_models_on_dataset(X_train_rand, y_L, X_test_rand, y_test)
    
    # MLP regression
    embedding_dim = X_train_rand.shape[1]
    dropout = 0
    regression_head = make_MLP_regression_head(embedding_dim,
                                               [cfg.layer1_dim, cfg.layer2_dim, cfg.layer3_dim],
                                               y_L, dropout, device)
    mlp_loss = evaluate_MLP_regressor(regression_head, X_train_rand, X_test_rand, y_L, y_test,
                                      cfg.regressor_epochs, cfg.lr_regressor, device)
    losses.append(mlp_loss)
    return losses, rf_model


def eval_flattened(X_L: np.ndarray, X_test: np.ndarray, y_L: np.ndarray, y_test: np.ndarray,
                   cfg, device: str) -> Tuple[List[float], object]:
    """Train models on fully flattened features and evaluate (includes MLP)."""
    X_train_flat = X_L.reshape(X_L.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0], -1)
    
    losses, rf_model = Preds().evaluate_models_on_dataset(X_train_flat, y_L, X_test_flat, y_test)
    
    # MLP regression
    embedding_dim = X_train_flat.shape[1]
    dropout = 0
    regression_head = make_MLP_regression_head(embedding_dim,
                                               [cfg.layer1_dim, cfg.layer2_dim, cfg.layer3_dim],
                                               y_L, dropout, device)
    mlp_loss = evaluate_MLP_regressor(regression_head, X_train_flat, X_test_flat, y_L, y_test,
                                      cfg.regressor_epochs, cfg.lr_regressor, device)
    losses.append(mlp_loss)
    return losses, rf_model
