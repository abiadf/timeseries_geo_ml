"""Central place to keep all paths to use in the project"""
from pathlib import Path

interim_data_loc = "../interim_data"
public_data_loc  = "../public_datasets"
encoders_folder  = "other_encoders"
ts2vec_params_loc= f"ts2vec_params"
asm_folder_loc   = "../public_datasets/3D/ASM"

# config paths
param_config_folder  = "param_config"
messager_yaml_path   = f"{param_config_folder}/messager.yaml"
data_params_yaml_path= f"{param_config_folder}/dataset_params.yaml"
params_path          = f"{param_config_folder}/baseline_params.yaml"
best_params_path     = f"{param_config_folder}/best_params.yaml"

# results
results_folder          = "results"
barlow_hyperparam_file  = f"{results_folder}/hyperparam_search_barlow.txt"
moment_hyperparam_file  = f"{results_folder}/hyperparam_search_moment.txt"
timevae_hyperparam_file = f"{results_folder}/hyperparam_search_timevae.txt"
ts2vec_hyperparam_file  = f"{results_folder}/hyperparam_search_ts2vec.txt"



