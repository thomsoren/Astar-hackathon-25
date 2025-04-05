import cv2
import os
from pathlib import Path

def yolo_to_xyxy(x_center, y_center, width, height, img_w, img_h):
    """
    Converts YOLO format (x_center, y_center, width, height) 
    to coordinates: (x1, y1, x2, y2). 
    All YOLO values are assumed to be normalized [0,1].
    """
    # Convert normalized coords to absolute pixel values
    x_center_abs = x_center * img_w
    y_center_abs = y_center * img_h
    w_abs = width * img_w
    h_abs = height * img_h

    # Compute top-left and bottom-right
    x1 = int(x_center_abs - (w_abs / 2))
    y1 = int(y_center_abs - (h_abs / 2))
    x2 = int(x_center_abs + (w_abs / 2))
    y2 = int(y_center_abs + (h_abs / 2))

    return x1, y1, x2, y2

def draw_bounding_boxes_from_yolo(image_path, label_path):
    """
    Given an image and its corresponding YOLO label file, 
    draw bounding boxes on the image and display it.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    h, w, _ = image.shape

    # If there's no label file, just show the image without boxes
    if not os.path.exists(label_path):
        print(f"No label file found for {image_path}. Showing image without boxes.")
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Objects", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Read YOLO label lines
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Usually YOLO line format: class x_center y_center width height
        parts = line.split()
        if len(parts) != 5:
            print(f"Skipping malformed line: {line}")
            continue

        cls_id = parts[0]
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, box_width, box_height, w, h)

        # Ensure bounding boxes are within image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        # Draw bounding box (green)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put the class label above the box
        cv2.putText(
            image, 
            f"Class {cls_id}", 
            (x1, max(0, y1 - 5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            2
        )

    # Create a resizable window to better view the image
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Get paths relative to script location
    script_dir = Path(__file__).parent
    IMAGES_FOLDER = script_dir / "data" / "images" / "train"
    LABELS_FOLDER = script_dir / "data" / "labels" / "train"

    # Get list of all images in the folder (filter by extension if needed)
    valid_exts = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(IMAGES_FOLDER) 
                   if f.lower().endswith(valid_exts)]

    for image_file in image_files:
        # Full path to the image
        image_path = os.path.join(IMAGES_FOLDER, image_file)

        # Label file has the same name, but with .txt extension
        # (if you store them differently, adjust here)
        label_basename = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(LABELS_FOLDER, label_basename)

        print(f"Processing image: {image_file}")
        draw_bounding_boxes_from_yolo(image_path, label_path)

if __name__ == "__main__":
    main()
