# Clean version without BOM
import cv2
import os
import yaml
from pathlib import Path

def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
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
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)['names']

def draw_bounding_boxes(image_path, label_path, class_names):
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
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, x, y, bw, bh = parts
            x1, y1, x2, y2 = yolo_to_xyxy(float(x), float(y), float(bw), float(bh), w, h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_names.get(int(cls_id), f"Class {cls_id}"),
                       (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    script_dir = Path(__file__).parent
    dataset_root = script_dir.parent.parent.parent / "data" / "yolo_dataset"
    class_names = load_class_names(dataset_root / "data.yaml")

    for img_file in (f for f in os.listdir(dataset_root / "images/train") 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))):
        img_path = dataset_root / "images/train" / img_file
        label_path = dataset_root / "labels/train" / f"{Path(img_file).stem}.txt"
        print(f"Processing {img_file}...")
        draw_bounding_boxes(img_path, label_path, class_names)

if __name__ == "__main__":
    main()
