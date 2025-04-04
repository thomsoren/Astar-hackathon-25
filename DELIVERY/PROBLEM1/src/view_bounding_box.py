﻿# -*- coding: utf-8 -*-
import cv2
import os
import yaml
from pathlib import Path

def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
    """Convert YOLO format to bounding box coordinates"""
    x_center_abs = x_center * img_w
    y_center_abs = y_center * img_h
    w_abs = width * img_w
    h_abs = height * img_h
    x1 = int(x_center_abs - (w_abs / 2))
    y1 = int(y_center_abs - (h_abs / 2))
    x2 = int(x_center_abs + (w_abs / 2))
    y2 = int(y_center_abs + (h_abs / 2))
    return x1, y1, x2, y2

def load_class_names(yaml_path):
    """Load class names from YOLO data.yaml file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def draw_bounding_boxes(image_path, label_path, class_names):
    """Draw bounding boxes with class names on image"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    h, w = image.shape[:2]
    
    if not os.path.exists(label_path):
        print(f"No label file found for {image_path}")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        return

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, x, y, bw, bh = parts
        x1, y1, x2, y2 = yolo_to_xyxy(
            float(x), float(y), float(bw), float(bh), w, h
        )

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display class name instead of just class ID
        class_name = class_names.get(int(cls_id), f"Class {cls_id}")
        cv2.putText(
            image, class_name, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Paths relative to script location
    script_dir = Path(__file__).parent
    dataset_root = script_dir.parent.parent.parent / "data" / "yolo_dataset"
    
    images_folder = dataset_root / "images" / "train"
    labels_folder = dataset_root / "labels" / "train"
    yaml_config = dataset_root / "data.yaml"

    class_names = load_class_names(yaml_config)

    for img_file in os.listdir(images_folder):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = images_folder / img_file
        label_path = labels_folder / f"{Path(img_file).stem}.txt"

        print(f"Processing {img_file}...")
        draw_bounding_boxes(img_path, label_path, class_names)

if __name__ == "__main__":
    main()
