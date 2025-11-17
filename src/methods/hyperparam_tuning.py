import __main__
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS


def make_trainable(X_train, X_test, y_train_scaled, y_test_scaled, device):
    """Creates a Ray-compatible trainable.
    Captures the dataset and device through closure."""
    def train_moment_ray(config):
        model_cfg = {
            "model_name":      config["model_name"],
            "task_name":       config["task_name"],
            "hidden_layers":   config["hidden_layers"],
            "unfreeze_last_n": config["unfreeze_last_n"],
            "dropout":         config["dropout"],}

        train_cfg = {
            "epochs":     config["epochs"],
            "batch_size": config["batch_size"],
            "lr_encoder": config["lr_encoder"],
            "lr_head":    config["lr_head"],
            "fine_tune":  config["fine_tune"],}

        # your existing full pipeline
        losses, r2, metrics = run_moment(
            X_train, X_test,
            y_train_scaled, y_test_scaled,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            device=device)

        test_rmse = losses[1]
        tune.report(test_rmse=test_rmse, r2=r2)
    return train_moment_ray


search_space = {
    "model_name": tune.choice(["moment-small", "moment-base"]),
    "task_name": tune.choice(["classification", "regression"]),
    "hidden_layers": tune.choice([[512,256,64,16],
                                  [256,128,32],
                                  [1024,512,128,32]]),
    "unfreeze_last_n": tune.choice([0, 1, 2, 3]),
    "epochs": tune.choice([5, 10, 15, 20]),
    "dropout": tune.choice([0.0, 0.5]),
    "batch_size": tune.choice([128, 256, 512]),
    "lr_encoder": tune.choice([1e-5, 1e-3]),
    "lr_head": tune.choice([1e-4, 1e-2]),
    "fine_tune": tune.choice([True, False]),}

trainable = make_trainable(X_train, X_test,
                           y_train_scaled, y_test_scaled,
                           device)
algo  = TuneBOHB(
    metric="test_rmse",  # the metric your trainable reports
    mode="min")           # "min" because lower RMSE is better
sched = HyperBandForBOHB(metric="test_rmse", mode="min")

analysis = tune.run(
    trainable,
    name="moment_bohb",
    resources_per_trial={"cpu": 4, "gpu": 1},
    num_samples=40,
    search_alg=algo,
    scheduler=sched,
    config=search_space,)

print("Best config:", analysis.best_config)

