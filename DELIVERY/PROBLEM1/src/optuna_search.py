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
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
IMG_SIZE = config["img_size"]
DATA_YAML = config["data_yaml"]
TEST_IMAGES_DIR = config["test_images_dir"]
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
    model = YOLO("yolo11m.pt")

    # Train with hyperparameters
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=f"optuna_trial_{trial.number}",
        plots=False,
        amp=True,
    # Early stopping parameter (uncomment if needed)
    # patience=10,  
        lr0=lr0,
        weight_decay=weight_decay,
        cos_lr=True,
        box=box,
        cls=cls,
        dfl=dfl,
        pose=pose,
        fraction=0.3
    )

    metrics = model.val(data=config["data_yaml"], split="val", imgsz=config["img_size"])
    
    mAP = metrics.box.map       # mAP50-95
    mAP50 = metrics.box.map50     # mAP50
    mAP75 = metrics.box.map75     # mAP75
    # Compute a composite score (adjust weights as needed)
    composite_score = (mAP + mAP50 + mAP75) / 3
    return composite_score

# # ===================================
# # 3. Run Optuna Hyperparameter Search
# # ===================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=75)  # Run 10 trials

# Get best trial results
best_trial = study.best_trial
best_params = best_trial.params

print(f"Best Trial: {best_trial.number}")
print(f"Best Hyperparameters: {best_params}")
