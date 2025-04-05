# PROJECT1: Image Augmentation with YOLO

## Description

This project focuses on augmenting images of supermarket products and training a YOLO model to detect these products. It includes scripts for data augmentation, training, and evaluation.

## Setup Instructions

1.  Clone the repository.
2.  Install the required dependencies using `pip install -r requirements.txt`.
3.  Download the YOLOv11m.pt pretrained model and place it in the `src/` directory.

## Usage Instructions

1.  Run `augment_pics.py` to augment the images.
2.  Run `train_yolo.py` to train the YOLO model.
3.  Use `view-bb.py` to visualize bounding boxes.

## File Structure

*   `src/`: Contains the source code for the project.
    *   `augment_pics.py`: Script for augmenting images.
    *   `train_yolo.py`: Script for training the YOLO model.
    *   `view-bb.py`: Script for visualizing bounding boxes.
    *   `config.yaml`: Configuration file for training.
    *   `plu_mapping.json`: Mapping of PLU codes to product names.
*   `data/`: Contains the augmented images.
*   `model/`: Contains the trained YOLO model.

## Dependencies

*   torch
*   torchvision
*   opencv-python
*   pyyaml
*   optuna

## Model

The `best.pt` model is located in the `model/` directory.

## Accuracy Plot

The `accuracy_plot.png` is located in the main directory.
