# ===================================
# 1. Installs and Imports
# ===================================
import os
import yaml
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch

torch.cuda.empty_cache()

# ===================================
# 2. Directory & Config Setup
# ===================================
CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

ROOT = os.getcwd()

# Extract configuration values
MODEL = config["model"]
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
IMG_SIZE = config["img_size"]
DATA_YAML = config["data_yaml"]
TEST_IMAGES_DIR = config["test_images_dir"]
# PATIENCE = config["patience"]

# W&B Settings
USE_WANDB = config["use_wandb"]
WANDB_PROJECT = config["wandb_project"]
WANDB_TEAM = config.get("wandb_team", None)
WANDB_RUN_NAME = config["wandb_run_name"]

# # ===================================
# # 2. Define Optuna Optimization
# # ===================================
def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    # Hyperparameter search space
    lr0 = 0.0010437818703644063
    weight_decay = 4.099532698065393e-05
    box  = trial.suggest_float("box",  5.0, 10.0)   # around 7.5
    cls  = trial.suggest_float("cls",  0.1, 1.0)    # around 0.5
    dfl  = trial.suggest_float("dfl",  1.0, 2.0)    # around 1.5
    pose = trial.suggest_float("pose", 8.0, 15.0)   # around 12.0

    # Initialize YOLO model
    model = YOLO(MODEL)

    # Train with hyperparameters
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=f"optuna_trial_{trial.number}",
        plots=False,
        amp=True,
        # patience=PATIENCE,  # Early stopping
        lr0=lr0,
        weight_decay=weight_decay,
        cos_lr=True,
        box=box,
        cls=cls,
        dfl=dfl,
        pose=pose,
        fraction=0.3
    )

    # Get validation mAP50-95 score (higher is better)
    val_metrics = model.val(data=DATA_YAML, split="val", imgsz=IMG_SIZE)
    val_map50_95 = float(val_metrics.box.map)
    

    return val_map50_95  # Optuna maximizes this

# # ===================================
# # 3. Run Optuna Hyperparameter Search
# # ===================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # Run 10 trials

# Get best trial results
best_trial = study.best_trial
best_params = best_trial.params
print(f"Best Trial: {best_trial.number}")
print(f"Best Hyperparameters: {best_params}")