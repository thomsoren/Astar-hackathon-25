#!/usr/bin/env python3
"""
annotation_tool.py

Tool to assist with semi-automated annotation of images.
If you already have YOLO model predictions, you can load them 
and allow the user to correct bounding boxes or class labels.
"""

import os
import cv2
import json

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# A simple data structure for bounding boxes
# Format: (class_id, x_center, y_center, width, height)
# in YOLO format: all coords are normalized [0..1] w.r.t image size
# If you store them in pixel values, adapt accordingly.

def load_or_create_annotations(annot_file):
    """
    Loads existing annotations if annot_file exists, otherwise returns an empty list/dict.
    """
    if os.path.exists(annot_file):
        with open(annot_file, 'r') as f:
            data = json.load(f)
        return data
    else:
        return []

def save_annotations(annot_file, annotations):
    """
    Save annotations to JSON or another format. 
    In a real YOLO workflow, you might want to save them in .txt YOLO format.
    """
    with open(annot_file, 'w') as f:
        json.dump(annotations, f, indent=2)

def predict_bboxes(model, img_path, conf_threshold=0.25):
    """
    Use the YOLO model to generate bounding boxes for an image.
    Return the bounding boxes in YOLO or pixel format as needed.
    """
    if not model:
        print("[ERROR] Model not available. Please install 'ultralytics' or load your model differently.")
        return []

    results = model.predict(img_path, conf=conf_threshold)
    if len(results) == 0:
        return []
    bboxes = []
    # Each result is a list of detections. We'll take the first result only for single image
    for det in results[0].boxes:
        xywh = det.xywh[0].cpu().numpy()  # (x, y, w, h) in pixels
        cls = int(det.cls[0].cpu().numpy())
        bboxes.append((cls, *xywh))  # store in pixel format
    return bboxes

def visualize_annotations(img, bboxes, class_names=None, color=(0, 255, 0)):
    """
    Draw bounding boxes on an image for visualization.
    bboxes expected as list of (cls_id, x, y, w, h) in pixel coordinates.
    """
    h_img, w_img = img.shape[:2]
    for bbox in bboxes:
        cls_id, x, y, w, h = bbox
        # Convert xywh -> top-left, bottom-right
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = str(cls_id) if class_names is None else class_names[cls_id]
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def annotate_images(img_dir, output_dir, model_weights=None, class_list=None, conf_threshold=0.25):
    """
    Main function to load images from img_dir, run YOLO predictions (optional),
    and let the user correct bounding boxes. 
    This example is CLI-based. A real tool might require a GUI.

    :param img_dir: Directory with images to annotate.
    :param output_dir: Where to store annotation files or revised images.
    :param model_weights: Path to YOLO weights, if you want to auto-generate bounding boxes. 
    :param class_list: A list or dict of class names for labeling. 
    :param conf_threshold: Confidence threshold for detection.
    """
    if not os.path.isdir(img_dir):
        print(f"[ERROR] Image directory does not exist: {img_dir}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Optional: load YOLO model
    model = None
    if model_weights and YOLO:
        print(f"[INFO] Loading YOLO model from {model_weights}")
        model = YOLO(model_weights)

    images = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png']]
    images.sort()

    for img_file in images:
        img_path = os.path.join(img_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Could not read image: {img_path}")
            continue
        
        # 1. Load or create annotation file
        base_name = os.path.splitext(img_file)[0]
        annot_file = os.path.join(output_dir, base_name + ".json")
        annotations = load_or_create_annotations(annot_file)  # can store in JSON

        # 2. If no existing annotations, optionally run YOLO predictions
        if len(annotations) == 0 and model is not None:
            bboxes = predict_bboxes(model, img_path, conf_threshold=conf_threshold)
            # Convert bounding boxes to your desired structure
            for (cls_id, x, y, w, h) in bboxes:
                # For example, store them as dict
                annotations.append({
                    "class_id": int(cls_id),
                    "x_center": float(x),
                    "y_center": float(y),
                    "width": float(w),
                    "height": float(h)
                })

        # 3. Display the image with bounding boxes so user can correct them
        #    (for CLI, we'll just show for a second or press a key. 
        #     A real tool might allow mouse-based editing.)
        vis_bboxes = []
        for ann in annotations:
            cls_id = ann["class_id"]
            x = ann["x_center"]
            y = ann["y_center"]
            w = ann["width"]
            h = ann["height"]
            vis_bboxes.append((cls_id, x, y, w, h))
        
        # Visualization
        display_img = image.copy()
        display_img = visualize_annotations(display_img, vis_bboxes, class_list)

        cv2.imshow("Annotation Tool - Press any key to continue, ESC to skip", display_img)
        key = cv2.waitKey(500)  # show for 500 ms
        if key == 27:  # ESC
            print("[INFO] Skipped editing annotations.")
            pass
        else:
            print("[INFO] No editing function implemented in this CLI example.")
            print("      In a real scenario, you'd open a bounding box editor here.")
        
        # 4. Save updated annotations
        save_annotations(annot_file, annotations)

    cv2.destroyAllWindows()

def main():
    """
    A simple command-line interface to run the annotation tool.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Semi-automatic annotation tool.")
    parser.add_argument('--img_dir', type=str, required=True, 
                        help='Directory containing images to annotate')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to store annotation files (JSON, YOLO labels, etc.)')
    parser.add_argument('--weights', type=str, default=None, 
                        help='Path to YOLO model weights (optional).')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for auto-detection.')
    parser.add_argument('--class_list', type=str, default=None,
                        help='Path to a text file with one class name per line, optional.')
    args = parser.parse_args()

    # Load class names if provided
    classes = None
    if args.class_list and os.path.exists(args.class_list):
        with open(args.class_list, 'r') as f:
            classes = [line.strip() for line in f.readlines()]

    annotate_images(
        img_dir=args.img_dir, 
        output_dir=args.output_dir, 
        model_weights=args.weights, 
        class_list=classes,
        conf_threshold=args.conf
    )

if __name__ == "__main__":
    main()