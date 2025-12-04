import __main__
import sys, os
project_root = os.path.abspath("..")  # adjust if notebook is elsewhere
sys.path.insert(0, project_root)
from typing import Dict, List, Literal, Tuple, Optional
import logging
import json
from pathlib import Path

import optuna
import torch

from benchmarks.ts2vec_runner import run_ts2vec, log_ts2vec_results
from benchmarks.timevae_runner import run_timevae, log_timevae_results

from utils.data_utils import convert_numpy
import param_config.config_paths as P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")


class OptunaTuner:
    "Hyperparameter tuner for various methods using Optuna"

    @staticmethod
    def timevae_optuna_objective(trial, X_train, X_test, y_train, y_test,
                        device, timevae_file_path, desired_dataset, log_file):
        latent_dim        = trial.suggest_categorical("latent_dim", [4, 8, 12, 16, 32])
        hidden_layers     = trial.suggest_categorical("hidden_layers", [[12,8,12], [32,16,8], [64,32,16], [64,64,64]])
        reconstruction_wt = trial.suggest_float("reconstruction_wt", 0.5, 3.5)
        batch_size        = trial.suggest_categorical("batch_size", [128])
        lr                = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        train_epochs      = 100

        model_cfg = {
            "latent_dim": latent_dim,
            "hidden_layers": hidden_layers,
            "reconstruction_wt": reconstruction_wt}
        train_cfg = {
            "train_epochs": train_epochs,
            "lr": lr,
            "batch_size": batch_size}
        try:
            losses, r2, _, _, recon_loss_test, _, _ = run_timevae(
                X_train, X_test, y_train, y_test,
                timevae_file_path=timevae_file_path,
                device=device,
                batch_size=train_cfg["batch_size"],
                train_epochs=train_cfg["train_epochs"],
                lr_training=train_cfg["lr"],
                latent_dim=model_cfg["latent_dim"],
                hidden_layer_sizes=model_cfg["hidden_layers"],
                reconstruction_wt=model_cfg["reconstruction_wt"],
                desired_dataset=desired_dataset,
                force_train=True,)

            record = convert_numpy({
                **model_cfg,
                **train_cfg,
                "rmse": float(losses[0]),
                "r2": float(r2),
                "recon_loss": float(recon_loss_test)})
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            return float(losses[0])

        except Exception as e:
            print("Trial failed:", e)
            return float("nan")

    @staticmethod
    def ts2vec_optuna_objective(trial):
        # Define hyperparameter search space
        hidden_dims = trial.suggest_categorical("hidden_dims", [12, 16, 20, 24])
        latent_dims = trial.suggest_categorical("latent_dims", [8, 12, 16, 20])
        depth       = trial.suggest_categorical("depth", [3, 4, 5])
        lr          = trial.suggest_categorical("lr", [0.001, 0.005])
        z_pooling   = trial.suggest_categorical("z_pooling_method", ["None"])#, "mean", "max"])
        batch_size  = 1024
        epochs      = 100

        model_cfg = {
            "hidden_dims": hidden_dims,
            "latent_dims": latent_dims,
            "depth": depth,
            "z_pooling_method": z_pooling}

        train_cfg = {
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": 25,
            "window_size": window_size} # use your predefined window_size
        try:
            losses, r2, metrics = run_ts2vec(X_train, X_test, y_train_scaled, y_test_scaled,
                                            model_cfg=model_cfg, train_cfg=train_cfg, device=device)
            # Log results
            record = {**model_cfg, **train_cfg, "test_rmse": losses[0], "r2": r2}
            with open(P.ts2vec_hyperparam_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            return losses[0]  # Optimize RMSE
        except Exception as e:
            print("Trial failed:", e)
            return float("inf")

