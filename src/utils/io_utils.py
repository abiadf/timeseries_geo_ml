from typing import Union, Generator, Tuple, Optional, List
import json
import os
import requests
import time
import yaml
import torch
import numpy as np
import pandas as pd

def read_yaml_params(file_path: str) -> dict:
    """Read parameters from a YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def clean_notebook(path: str) -> None:
    """Remove all outputs and execution counts from a .ipynb file."""
    with open(path) as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
    with open(path, "w") as f:
        json.dump(nb, f, indent=2)

def set_all_rand_seeds(seed: int) -> None:
    """set deterministic RNG for python / numpy / torch"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def split_npy_file(file_path: str, output_prefix: str, num_splits: int):
    """Splits a large .npy file along the first axis into multiple smaller .npy files.
    Args:
        file_path: Path to the original .npy file.
        output_prefix: Prefix for the output files.
        num_splits: Number of splits to create."""
    data            = np.load(file_path, mmap_mode='r')  # memory-map so we don't load all into RAM
    total_pages     = data.shape[0]
    pages_per_split = total_pages // num_splits
    remainder       = total_pages % num_splits

    start = 0
    for i in range(num_splits):
        end   = start + pages_per_split + (1 if i < remainder else 0)
        np.save(f"{output_prefix}_{i:03d}.npy", data[start:end])
        start = end



class Notifiers:
    @staticmethod
    def send_discord_message(webhook_url: str, message: str) -> None:
        "send discord message via webhook"
        data = {"content": message}
        r    = requests.post(webhook_url, json=data)
        r.raise_for_status()

    @staticmethod
    def make_beep_sound(times=1, delay=0.2):
        for _ in range(times):
            os.system('afplay /System/Library/Sounds/Blow.aiff')
            time.sleep(delay)


class JSONLogger:
    @staticmethod
    def load_json_file_safely(file_path) -> dict:
        """Load JSON file or return empty dict if missing/empty/invalid."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return {}
        try:
            return json.load(open(file_path))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def log_result_to_json(dataset: str, method: str, values: list[float], file_location: str, result_type: str = "metrics"):
        """Append one run's list/tuple of metrics for a dataset + method."""
        data = JSONLogger.load_json_file_safely(file_location)
        data.setdefault(dataset, {}).setdefault(result_type, {}).setdefault(method, []).append(list(values))
        with open(file_location, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def safe_call(func, *args, **kwargs):
        """Run func safely; swallow all exceptions."""
        try:
            func(*args, **kwargs)
        except Exception as e:
            args_repr = tuple(repr(a) for a in args)
            kwargs_repr = {k: repr(v) for k, v in kwargs.items()}
            print(f"[log skipped] {e} | args={args_repr} kwargs={kwargs_repr}")

    @staticmethod
    def X_summarize_runs_to_latex(methods: dict, num_runs: int) -> pd.DataFrame:
        """Return a DataFrame with LaTeX-ready mean±std for the last full block of N runs per method."""
        rows = []
        for method, runs in methods.items():
            total = len(runs)
            # Must have at least N runs, and must be divisible by N
            if total < num_runs or total % num_runs != 0:
                continue
            # Use the LAST COMPLETE BLOCK
            block = runs[-num_runs:] # safe because divisible by N
            arr   = np.array(block)  # shape (N, num_metrics)
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0)
            row   = {"method": method}
            for i, (m, s) in enumerate(zip(means, stds)):
                row[f"metric_{i}"] = f"\\val{{{m:.3f}}}{{{s:.3f}}}"
            rows.append(row)
        return pd.DataFrame(rows).fillna("--")

    @staticmethod
    def XX_summarize_runs_to_latex(methods: dict, num_runs: int, dataset: str) -> pd.DataFrame:
        """Return a DataFrame with LaTeX-ready mean±std for the last full block of N runs per method."""
        rows = []
        for method, runs in methods.items():
            total = len(runs)
            if total < num_runs or total % num_runs != 0:
                continue
            block = runs[-num_runs:]
            arr   = np.array(block)
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0)
            row   = {"dataset": dataset, "method": method}  # add dataset column
            for i, (m, s) in enumerate(zip(means, stds)):
                row[f"metric_{i}"] = f"\\val{{{m:.3f}}}{{{s:.3f}}}"
            rows.append(row)
        return pd.DataFrame(rows).fillna("--")

    @staticmethod
    def summarize_runs_to_latex(methods: dict, num_runs: int, dataset: str) -> pd.DataFrame:
        """Return a DataFrame with LaTeX-ready mean±std for the last full block of N runs per method,
        with dataset name only on the first row."""
        rows = []
        first_row = True
        for method, runs in methods.items():
            total = len(runs)
            if total < num_runs or total % num_runs != 0:
                continue
            block = runs[-num_runs:]
            arr   = np.array(block)
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0)
            row   = {"dataset": dataset if first_row else "", "method": method}  # only first row
            for i, (m, s) in enumerate(zip(means, stds)):
                row[f"metric_{i}"] = f"\\val{{{m:.3f}}}{{{s:.3f}}}"
            rows.append(row)
            first_row = False  # only show dataset in the first row of the group
        return pd.DataFrame(rows).fillna("--")
