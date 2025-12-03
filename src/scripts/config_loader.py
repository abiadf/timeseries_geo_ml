import os
import __main__
import numpy as np
from types import SimpleNamespace

from src.utils.io_utils import read_yaml_params 
import src.param_config.config_paths as P

def load_project_configuration(params_path, data_params_path, messager_path):
    """Load global config, apply env overrides, construct namespace."""
    params          = read_yaml_params(params_path)
    data_params     = read_yaml_params(data_params_path)
    messager_params = read_yaml_params(messager_path)

    # Dataset selection override via env
    env_dataset = os.getenv("DATASET")
    if env_dataset:
        params["basics"]["dataset"] = env_dataset

    desired_dataset = params["basics"].get("dataset") or params["basics"]["desired_dataset"]

    # Window length override via env
    dataset_window  = int(os.environ.get("WINDOW_LEN", data_params["general"]["window_len"]))

    # Seed logic
    if params["basics"]["freeze_rand_seed"]:
        rand_seed = params["basics"]["random_seed"]
    else:
        rand_seed = np.random.default_rng().integers(0, 10_000)

    cfg = SimpleNamespace(
        desired_dataset= desired_dataset,
        dataset_window = dataset_window,
        rand_seed      = rand_seed,
        params         = params,
        data_params    = data_params,

        # Paths
        interim_data_loc = P.interim_data_loc,
        public_data_loc  = P.public_data_loc,
        asm_folder_loc   = P.asm_folder_loc,
        webhook_url      = messager_params.get("webhook_url"),

        # Data config
        num_pages_to_use  = data_params["general"]["num_pages_to_use"],
        num_rows_per_page = data_params[desired_dataset]["num_rows_per_page"],
        windows_per_page  = data_params[desired_dataset]["num_window_splits"],
        train_epochs      = data_params["general"]["train_epochs"],

        # Basics
        label_frac    = params["basics"]["label_frac"],
        data_splitting= params["basics"]["data_splitting"],
        do_we_scale_y = params["basics"]["do_we_scale_y"],
        num_runs      = params["basics"]["num_runs"],

        # Regressor
        predictor_epochs= params["general_params"]["regressor"]["epochs_regressor"],
        layer1_dim      = params["general_params"]["regressor"]["layer1_dim"],
        layer2_dim      = params["general_params"]["regressor"]["layer2_dim"],
        layer3_dim      = params["general_params"]["regressor"]["layer3_dim"],
        lr_regressor    = params["general_params"]["regressor"]["lr"],
        regressor_epochs= params["general_params"]["regressor"]["epochs_regressor"],

        # CellSup / AE
        AE_lr       = params["cellsup"]["AE_lr"],
        weight_decay= params["cellsup"]["weight_decay"],
        dropout     = params["cellsup"]["dropout"],
        swav_iters  = params["cellsup"]["swav_iters"],
        swav_temp   = params["cellsup"]["swav_temp"],
        cluster_min = params["cellsup"]["clustering"]["cluster_min"],
        cluster_max = params["cellsup"]["clustering"]["cluster_max"],
        # Notebook check
        running_in_notebook=not hasattr(__main__, "__file__"))

    print(f"Config Loaded: Dataset={cfg.desired_dataset}, Window={cfg.dataset_window}, Seed={cfg.rand_seed}")
    return cfg

def load_specific_method_params(dataset: str, method: str, best_params: dict, dataset_params: dict) -> dict:
    """Merge generic method defaults + dataset-specific override."""
    merged = {}

    # 1. generic defaults for the method
    if method in dataset_params:
        merged.update(dataset_params[method])

    # 2. dataset–specific overrides
    if dataset in best_params and method in best_params[dataset]:
        print(f"Found best params for ({dataset}+{method}), overriding...")
        merged.update(best_params[dataset][method])
    return merged

