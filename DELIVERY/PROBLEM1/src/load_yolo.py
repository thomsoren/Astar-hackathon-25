# ===================================
# 1. Installs and Imports
# ===================================
import os
import yaml
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results
    
import torch
torch.cuda.empty_cache() 


# ===================================
# 2. Directory & Config Setup
# ===================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

ROOT = os.getcwd()

# Extract configuration values
EPOCHS = config["epochs"]
BATCH_SIZE = config["batch_size"]
IMG_SIZE = config["img_size"]
DATA_YAML = config["data_yaml"]
TEST_IMAGES_DIR = config["test_images_dir"]

# ===================================
# 3. Load Pretrained Model
# ===================================
model = YOLO('yolo11m.pt')

# ===================================
# 4. Train the Model
# ===================================
# Train
train_results = model.train(
   data=DATA_YAML,
   epochs=EPOCHS,
   imgsz=IMG_SIZE,
   batch=BATCH_SIZE,
   plots=True,
   amp=True
)

# ===================================
# 5. Evaluate on Validation
# ===================================
val_metrics = model.val(
   data=DATA_YAML,
   split="val",  # or "train" if you want to check training set performance
   imgsz=IMG_SIZE
)

# ===================================
# 6. Predict on Test Images
# ===================================
results = model.predict(
   source=TEST_IMAGES_DIR,
   imgsz=IMG_SIZE,
)

# Visualize the results
# Visualize the results
for result in results:
   result.show()
