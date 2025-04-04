# ===================================
# 1. Installs and Imports
# ===================================
!pip install ultralytics pyyaml wandb --quiet
import os
import yaml
import wandb
import pandas as pd
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

# W&B Settings
USE_WANDB = config["use_wandb"]
WANDB_PROJECT = config["wandb_project"]
WANDB_TEAM = config.get("wandb_team", None)
WANDB_RUN_NAME = config["wandb_run_name"]

# Initialize W&B
if USE_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_TEAM,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE
        }
    )

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
   name=WANDB_RUN_NAME,
   plots=True,
   amp=True
)

# For training metrics
if USE_WANDB:
    # Extract numeric values from metrics
    train_loss = float(train_results.box.loss) if hasattr(train_results, "box") and hasattr(train_results.box, "loss") else None
    train_map50 = float(train_results.box.map50) if hasattr(train_results, "box") and hasattr(train_results.box, "map50") else None
    train_map50_95 = float(train_results.box.map) if hasattr(train_results, "box") and hasattr(train_results.box, "map") else None

    # Log metrics to W&B
    wandb.log({
        "train_loss": train_loss,
        "train_map50": train_map50,
        "train_map50-95": train_map50_95,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE
    })

# ===================================
# 5. Evaluate on Validation
# ===================================
val_metrics = model.val(
   data=DATA_YAML,
   split="val",  # or "train" if you want to check training set performance
   imgsz=IMG_SIZE
)

# For validation metrics
if USE_WANDB:
    # Extract numeric values from metrics
    val_loss = float(val_metrics.box.loss) if hasattr(val_metrics, "box") and hasattr(val_metrics.box, "loss") else None
    val_map50 = float(val_metrics.box.map50) if hasattr(val_metrics, "box") and hasattr(val_metrics.box, "map50") else None
    val_map50_95 = float(val_metrics.box.map) if hasattr(val_metrics, "box") and hasattr(val_metrics.box, "map") else None

    # Log validation metrics
    wandb.log({
        "val_loss": val_loss,
        "val_map50": val_map50,
        "val_map50-95": val_map50_95
    })

# ===================================
# 6. Predict on Test Images
# ===================================
results = model.predict(
   source=TEST_IMAGES_DIR,
   imgsz=IMG_SIZE,
)

if USE_WANDB:
    for result in results:
        # Get the original image from the result
        img = result.orig_img  # Use orig_img instead of imgs[0]
        
        # Plot the predictions on the image
        plotted_img = result.plot()  # This returns the image with predictions drawn on it
        
        # Log to W&B
        wandb.log({"sample_prediction": wandb.Image(plotted_img)})

# Visualize the results
for result in results:
   result.show()



    #def run_yolo():
    # Load pre-trained YOLOv11 nano model
    #model = YOLO('yolo11m.pt')  # You can replace with yolo11s.pt, yolo11m.pt, etc.

    # Run inference on an image
    #results = model('path/to/your/image.jpg')  # replace with actual image path

    # Show results (bounding boxes, labels)
    #results.show()

if __name__ == "__main__":
    run_yolo()