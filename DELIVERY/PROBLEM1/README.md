# YOLOv11 Product Detection and Tracking System

## Description

This repository contains a computer vision solution utilizing YOLOv11 to identify and track products from top-view checkout videos. The system supports receipt generation and theft detection by accurately detecting and classifying items.

## Approach

Our solution leverages YOLOv11, an advanced object detection model, fine-tuned for our specific product classes. We incorporated a comprehensive pipeline from dataset preparation and augmentation to hyperparameter optimization and model training.

### Workflow:
1. **Data Preparation**
   - Convert raw annotations to YOLO format using a custom script.
   - Augment data to ensure a balanced and robust dataset.

2. **Model Training**
   - Conduct hyperparameter optimization using Uptuna.
   - Train YOLOv11 with optimized parameters.

3. **Evaluation**
   - Evaluate on a validation dataset, achieving strong detection performance.

## Setup and Usage

### Installation
Clone the repository and install dependencies:

```bash
git clone [repo-link]
cd [repo-name]
pip install -r requirements.txt
```

### Data Preparation
Convert your dataset to YOLO format:

```bash
cd src
python prepare_yolo_dataset.py  # Ensure file paths are correctly configured
```

### Data Augmentation
Augment the dataset to achieve balanced classes (minimum 150 images per class):

```bash
python augment.py  # Adjust augmentation parameters directly in the script as needed
```

### Bounding Box Visualization
Validate bounding box annotations visually:

```bash
python view_bb.py  # Ensure paths are correctly set to your dataset
```

### Hyperparameter Optimization
Run hyperparameter optimization:

```bash
python uptuna_search.py
```

### Model Training
Train YOLOv11 with optimized parameters:

```bash
python train_yolo.py
```

Example optimized parameters:

```python
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
```

## Data Augmentation Details
The augmentation pipeline employed includes:

```python
augmentor = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.3),
    A.MotionBlur(blur_limit=3, p=0.1),
    A.Affine(scale=(0.9, 1.1), rotate=(-5, 5), p=0.2),
])
```

## Known Limitations and Improvements

- Performance may vary under significantly different lighting or camera angles.
- Additional training data could further enhance detection accuracy.
- Real-time performance optimizations could be explored.

## Running the Validation

To run validation, use:

```bash
python main.py --val_dir /path/to/val_folder
```

## Results
Our final model achieved excellent performance on the validation set, demonstrating high accuracy across classes.

---

For further assistance or collaboration, please contact the maintainers.

