from typing import Union, Generator, Tuple, Optional, List
import json
import os
import requests
import time
import yaml

import numpy as np
import torch

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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    def load_json_file_safely(file_path: str) -> dict:
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
        # data.setdefault(dataset, {}).setdefault(method, []).append(list(values))
        # json.dump(data, open(file_location, "w"), indent=2)
        data.setdefault(dataset, {}).setdefault(result_type, {}).setdefault(method, []).append(list(values))
        with open(file_location, "w") as f:
            json.dump(data, f, indent=2)

