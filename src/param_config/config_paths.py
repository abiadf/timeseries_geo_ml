"""Central place to keep all paths to use in the project"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
SRC_ROOT     = Path(__file__).parent.parent.resolve()

encoders_folder  = SRC_ROOT / "other_encoders"
ts2vec_params_loc= SRC_ROOT / "ts2vec_params"

# data paths
interim_data_loc = PROJECT_ROOT / "interim_data"
public_data_loc  = PROJECT_ROOT / "public_datasets"
asm_folder_loc   = PROJECT_ROOT / "public_datasets/3D/ASM"

# config paths
param_config_folder  = SRC_ROOT / "param_config"
messager_yaml_path   = param_config_folder / "messager.yaml"
data_params_yaml_path= param_config_folder / "dataset_params.yaml"
params_path          = param_config_folder / "baseline_params.yaml"
best_params_path     = param_config_folder / "best_params.yaml"

# results
results_folder          = SRC_ROOT / "results"
barlow_hyperparam_file  = results_folder / "hyperparam_search_barlow.txt"
moment_hyperparam_file  = results_folder / "hyperparam_search_moment.txt"
timevae_hyperparam_file = results_folder / "hyperparam_search_timevae.txt"
ts2vec_hyperparam_file  = results_folder / "hyperparam_search_ts2vec.txt"
results_file            = results_folder / "results_numbers.json"
latex_results_file      = results_folder / "latex_results.txt"

#hyperparam search results files
hyperparam_search_folder= results_folder / "hyperparam_search"
timevae_hyperparam_file = hyperparam_search_folder / "timevae.txt"
ts2vec_hyperparam_file  = hyperparam_search_folder / "ts2vec.txt"
moment_hyperparam_file  = hyperparam_search_folder / "moment.txt"
barlow_hyperparam_file  = hyperparam_search_folder / "barlow.txt"

