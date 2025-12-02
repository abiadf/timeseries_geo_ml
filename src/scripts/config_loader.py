import os
import __main__
import numpy as np
from types import SimpleNamespace
from utils.io_utils import read_yaml_params 
from param_config.config_paths import interim_data_loc, public_data_loc, asm_folder_loc

def load_project_configuration(params_path, data_params_path, messager_path):
    """Also applies random seed"""
    # 1. Load YAMLs
    params          = read_yaml_params(params_path)
    data_params     = read_yaml_params(data_params_path)
    messager_params = read_yaml_params(messager_path)

    # 2. Handle Logic: Dataset Override (Env Var)
    env_dataset = os.getenv("DATASET")
    if env_dataset:
        params["basics"]["dataset"] = env_dataset
    
    desired_dataset = params["basics"].get("dataset") or params["basics"]["desired_dataset"]

    # 3. Handle Logic: Window Override
    if "WINDOW_LEN" in os.environ:
        dataset_window = int(os.environ["WINDOW_LEN"])
    else:
        dataset_window = int(data_params["general"]["window_len"])

    # 4. Handle Logic: Random Seed
    if params["basics"]["freeze_rand_seed"]:
        rand_seed = params["basics"]["random_seed"]
    else:
        rng       = np.random.default_rng()
        rand_seed = rng.integers(0, 10_000)

    # 5. Pack variables into the Namespace
    cfg = SimpleNamespace(
        # --- Core ---
        desired_dataset= desired_dataset,
        dataset_window = dataset_window,
        rand_seed      = rand_seed,
        params         = params,          # Keep full dict just in case
        data_params    = data_params,     # Keep full dict just in case
        
        # --- Paths (Hardcoded defaults based on your snippet) ---
        interim_data_loc= interim_data_loc,
        public_data_loc = public_data_loc,
        asm_folder_loc  = asm_folder_loc,
        webhook_url     = messager_params.get("webhook_url"),

        # --- Data Config ---
        num_pages_to_use  = data_params["general"]["num_pages_to_use"],
        num_rows_per_page = data_params[desired_dataset]["num_rows_per_page"],
        windows_per_page  = data_params[desired_dataset]["num_window_splits"],
        train_epochs      = data_params["general"]["train_epochs"],
        
        # --- Basics ---
        label_frac    = params["basics"]["label_frac"],
        data_splitting= params["basics"]["data_splitting"],
        do_we_scale_y = params["basics"]["do_we_scale_y"],
        num_runs      = params["basics"]["num_runs"],
        
        # --- Model / Regressor ---
        predictor_epochs = params["general_params"]["regressor"]["epochs_regressor"],
        layer1_dim       = params["general_params"]["regressor"]["layer1_dim"],
        layer2_dim       = params["general_params"]["regressor"]["layer2_dim"],
        layer3_dim       = params["general_params"]["regressor"]["layer3_dim"],
        lr_regressor     = params["general_params"]["regressor"]["lr"],
        regressor_epochs = params["general_params"]["regressor"]["epochs_regressor"],

        # --- CellSup / AE ---
        AE_lr        = params["cellsup"]["AE_lr"],
        weight_decay = params["cellsup"]["weight_decay"],
        dropout      = params["cellsup"]["dropout"],
        swav_iters   = params["cellsup"]["swav_iters"],
        swav_temp    = params["cellsup"]["swav_temp"],
        cluster_min  = params["cellsup"]["clustering"]["cluster_min"],
        cluster_max  = params["cellsup"]["clustering"]["cluster_max"],
        
        # --- Env Check ---
        running_in_notebook=not hasattr(__main__, "__file__")
    )
    
    print(f"Config Loaded: Dataset={cfg.desired_dataset}, Window={cfg.dataset_window}, Seed={cfg.rand_seed}")
    return cfg
