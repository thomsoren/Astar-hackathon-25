# ===================================
# 1. Installs and Imports
# ===================================
import os
import yaml
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results
import time
import matplotlib.pyplot as plt
import seaborn as sns
    
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

model.train(
    data=DATA_YAML,
    epochs=400,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name="best_model_lol",
    plots=True,
    amp=True,
    patience=20,
    lr0=0.0010437818703644063,
    weight_decay=0.00004099532698065393,
    box=7.18752296146375,
    cls=0.34256017682754214,
    dfl=1.5887712944378107,
    pose=8.392640010346742,
    cos_lr=True
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

# ===================================
# 6. Visualize Training Progress
# ===================================
# This code reads the CSV logged by YOLO and creates custom plots

# Path to the YOLO results CSV (adjust if needed)
results_csv = os.path.join(ROOT, "runs", "detect", "train", "results.csv")

# (Optional) Wait for results.csv to exist (avoid FileNotFoundError)
while not os.path.exists(results_csv):
    print("Waiting for results.csv to be generated...")
    time.sleep(10)

# Read CSV
df = pd.read_csv(results_csv)
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces

# Create subplots using seaborn
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

# Plot columns
sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4,1])

# Set subplot titles
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')

# Add a suptitle
plt.suptitle('Training Metrics and Loss', fontsize=24)

# Adjust layout to make space for suptitle
plt.subplots_adjust(top=0.85)
plt.tight_layout()

# Show the plot
plt.show()
