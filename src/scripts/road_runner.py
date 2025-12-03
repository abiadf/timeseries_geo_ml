"""Run this from src, ie: 'python -m src.scripts.road_runner.py' """
import __main__
import logging
import yaml
from datetime import datetime
import numexpr as ne # makes numpy operations faster
import category_encoders as ce
import numpy as np

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(torch.cuda.memory_reserved(0) / 1e6, "MB reserved")
    print(torch.cuda.memory_allocated(0) / 1e6, "MB allocated")

from src.scripts.timevae_script import run_timevae_block
from src.scripts.ts2vec_script import run_ts2vec_block
from src.scripts.moment_script import run_moment_block
from src.scripts.barlow_cnn_script import run_barlow_cnn_block
from src.scripts.config_loader import load_project_configuration, load_specific_method_params
from src.scripts.pipeline import load_the_data, split_data_to_labeled_unlabeled
from src.scripts.direct_preds_script import eval_mean_row, eval_custom_row, eval_random_row, eval_flattened

from src.utils.io_utils import JSONLogger, Notifiers, read_yaml_params, set_all_rand_seeds
import src.param_config.config_paths as P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logging.info("Starting process...")
logging.warning("Something looks off...")
logging.error("Something failed.")

# %%
"[RUN ME] Setup step"
cfg         = load_project_configuration(P.params_path, P.data_params_yaml_path, P.messager_yaml_path)
params      = cfg.params
data_params = cfg.data_params

set_all_rand_seeds(cfg.rand_seed)

X_train, X_test, y_train_scaled, y_test_scaled, window_size = load_the_data(
    cfg.desired_dataset,
    cfg.num_pages_to_use,
    cfg.do_we_scale_y,
    cfg.dataset_window,
    cfg.rand_seed,
    cfg.num_rows_per_page,
    cfg.params,
    cfg.label_frac,
    use_cache=False)
X_L, y_L, X_U, y_U, y_train_scaled, y_test_scaled, timevae_file_path = split_data_to_labeled_unlabeled(
    cfg.desired_dataset,
    P.interim_data_loc,
    cfg.data_splitting,
    cfg.label_frac,
    X_train, y_train_scaled,
    X_test, y_test_scaled,
    cfg.params,
    rand_seed=cfg.rand_seed)

# %%
"script runs"
best_params = yaml.safe_load(open(P.best_params_path))

if params["run_console"]["timevae"]:
    timevae_raw = load_specific_method_params(dataset=cfg.desired_dataset, method="timevae",
                                              best_params=best_params, dataset_params=data_params)
    timevae_cfg = {"timevae": timevae_raw}
    timevae_losses, timevae_recon_loss_test, r2, metrics, model_cfg, train_cfg = run_timevae_block(
        X_train, X_test, y_train_scaled, y_test_scaled,
        timevae_file_path,
        params=timevae_cfg,
        desired_dataset=cfg.desired_dataset,
        window_size=window_size,
        device=device,
        force_train=False)

if params["run_console"]["ts2vec"]:
    ts2vec_raw = load_specific_method_params(dataset=cfg.desired_dataset, method="ts2vec",
                                      best_params=best_params, dataset_params=data_params)
    ts2vec_cfg = {"ts2vec": ts2vec_raw}
    ts2vec_losses, r2, metrics, model_cfg, train_cfg = run_ts2vec_block(
        X_train, X_test, y_train_scaled, y_test_scaled,
        ts2vec_cfg,
        cfg.desired_dataset,
        window_size,
        device)

if params["run_console"]["moment"]:
    moment_raw = load_specific_method_params(dataset=cfg.desired_dataset, method="moment",
                                      best_params=best_params, dataset_params=data_params)
    moment_cfg = {"moment": moment_raw}
    moment_losses, r2, metrics, model_cfg, train_cfg = run_moment_block(
        X_train, X_test, y_train_scaled, y_test_scaled,
        moment_cfg,
        cfg.desired_dataset,
        device)

if params["run_console"]["barlow_cnn"]:
    barlow_raw = load_specific_method_params(dataset=cfg.desired_dataset, method="barlow_cnn",
                                      best_params=best_params, dataset_params=data_params)
    barlow_cfg = {"barlow_cnn": barlow_raw}
    barlow_cnn_losses, r2, metrics, barlow_recon_train, barlow_recon_test, model_cfg, train_cfg = run_barlow_cnn_block(
        X_train, X_test, y_train_scaled, y_test_scaled,
        barlow_cfg,
        desired_dataset=cfg.desired_dataset,
        device=device)

if params["run_console"]["direct_preds"]["mean_X"]:
    mean_losses, rf_model_mean = eval_mean_row(X_L, X_test, y_L, y_test_scaled, cfg, device)

if params["run_console"]["direct_preds"]["custom_row"]:
    custom_row_number = -1
    custom_losses, rf_model_custom = eval_custom_row(X_L, X_test, y_L, y_test_scaled, cfg, device, custom_row_number)

if params["run_console"]["direct_preds"]["random_row"]:
    rand_losses, rf_model_rand = eval_random_row(X_L, X_test, y_L, y_test_scaled, cfg, device)

if params["run_console"]["direct_preds"]["flatten_X"]:
    flat_losses, rf_model_flat = eval_flattened(X_L, X_test, y_L, y_test_scaled, cfg, device)

# %%
"Saving to file"

if params["run_console"]["direct_preds"]["mean_X"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "mean(X)", mean_losses, P.results_file, result_type="rmse")
if params["run_console"]["direct_preds"]["flatten_X"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "flattened(X)", flat_losses, P.results_file, result_type="rmse")
if params["run_console"]["direct_preds"]["random_row"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "random_row(X)", rand_losses, P.results_file, result_type="rmse")
if params["run_console"]["direct_preds"]["custom_row"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "custom_row(X)", custom_losses, P.results_file, result_type="rmse")

if params["run_console"]["timevae"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "TimeVAE", timevae_losses, P.results_file, result_type="rmse")
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "TimeVAE", [timevae_recon_loss_test], P.results_file, result_type="l_recons")
    # JSONLogger.log_result_to_json(cfg.desired_dataset, "TimeVAE", [timevae_profiling_metrics], P.results_file, result_type="profiling")
if params["run_console"]["ts2vec"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "TS2Vec", ts2vec_losses, P.results_file, result_type="rmse")
if params["run_console"]["ts2vec_fed"] == True:
    JSONLogger.log_result_to_json(cfg.desired_dataset, "TS2Vec (fed)", ts2vec_fed_losses, P.results_file, result_type="rmse")
if params["run_console"]["moment"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "Moment (cent)", moment_losses, P.results_file, result_type="rmse")
if params["run_console"]["moment_fed"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, f"Moment (fed, dim={dim_splitting})", moment_fed_losses, P.results_file, result_type="rmse")
if params["run_console"]["cellsup"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "Cellsup", cellsup_losses, P.results_file, result_type="rmse")
if params["run_console"]["barlow_cnn"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "Barlow (CNN)", barlow_cnn_losses, P.results_file, result_type="rmse")
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "Barlow (CNN)", [barlow_recon_test], P.results_file, result_type="rmse")
if params["run_console"]["cnn_lstm"] == True:
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "LSTM (X)", lstm_losses, P.results_file, result_type="rmse")
    JSONLogger.safe_call(JSONLogger.log_result_to_json, cfg.desired_dataset, "CNN (X)", cnn_mean_losses, P.results_file, result_type="rmse")

# ---- Read JSON ----
data = JSONLogger.load_json_file_safely(P.results_file)
methods_by_type = data.get(cfg.desired_dataset, {})

for result_type, methods in methods_by_type.items():
    ready_methods = {}
    for method, runs in methods.items():
        if len(runs) >= cfg.num_runs and len(runs) % cfg.num_runs == 0:
            ready_methods[method] = runs
            print(f"Added {cfg.desired_dataset} / {method} / {result_type} to latex results")
    if ready_methods:
        timestamp = datetime.now().strftime("%H:%M")
        with open(P.latex_results_file, "a") as f:
            f.write(f"-- {cfg.desired_dataset} {timestamp} {cfg.data_splitting=} {cfg.label_frac=} {cfg.dataset_window=} {result_type=} --\n")
            for method, runs in ready_methods.items():
                arr = np.array(runs[-cfg.num_runs:])
                means, stds = arr.mean(axis=0), arr.std(axis=0)
                line = f"{method} & " + " & ".join(f"\\val{{{m:.3f}}}{{{std:.3f}}}" for m, std in zip(means, stds)) + "\n"
                f.write(line)

Notifiers.send_discord_message(cfg.webhook_url, "Run finished")
# Notifiers.make_beep_sound(times=3, delay=0.2)
