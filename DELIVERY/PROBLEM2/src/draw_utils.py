#!/usr/bin/env python3
"""
draw_utils.py

Utility functions for drawing bounding boxes, labels, or other overlays on images.
"""

import cv2

def draw_bboxes(image, bboxes, class_names=None, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes on an image.

    :param image: The image on which to draw (numpy array in BGR format).
    :param bboxes: A list of bounding boxes. Each bounding box can be in either:
                   1) pixel coordinates: (x1, y1, x2, y2)  <-- top-left and bottom-right
                   2) YOLO xywh format: (x_center, y_center, width, height).
                   The function will need to know which format you're using.
    :param class_names: (Optional) If you want to include class labels,
                        pass a list/dict of class name strings, or None if not needed.
    :param color: The color used for bounding boxes and text (B,G,R).
    :param thickness: Line thickness for boxes/rectangles.
    :return: The image with bounding boxes drawn.
    """

    # Assume each item in bboxes is a tuple or list like:
    #   (cls_id, x1, y1, x2, y2) for pixel coords, or 
    #   (cls_id, x_center, y_center, w, h) for YOLO style.
    # You can unify or detect the format as needed.
    for bbox in bboxes:
        if len(bbox) == 5:
            # If using pixel-based bounding boxes, might look like: (cls_id, x1, y1, x2, y2)
            # OR YOLO-based: (cls_id, x, y, w, h)
            # Let's guess if x2 < x1, or something is off, etc. 
            # In practice, define a consistent format for your pipeline.

            cls_id = int(bbox[0])
            x1, y1, x2, y2 = convert_bbox_format(bbox)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # If class_names is provided and index is valid, draw label
            if class_names and cls_id < len(class_names):
                label_text = class_names[cls_id]
                cv2.putText(image, label_text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image

def convert_bbox_format(bbox, image_shape=None):
    """
    Convert from YOLO (cls_id, x_center, y_center, w, h) to pixel coords
    or confirm already in pixel coords. This is a minimal example.

    :param bbox: e.g. (cls_id, x_center, y_center, w, h) 
                 OR (cls_id, x1, y1, x2, y2).
    :param image_shape: (height, width), if needed for scaling YOLO coords.
    :return: (x1, y1, x2, y2) in pixel coords.
    """

    # If you always store bounding boxes in pixel coords,
    # simply return them. Otherwise, detect YOLO format and convert.
    cls_id = bbox[0]
    coords = bbox[1:]  # The rest
    if len(coords) == 4:
        # Heuristically check if this is YOLO format or already pixel corners
        # YOLO format typically has smaller x_center, y_center, w, h if they're normalized.
        # So, if e.g. x_center <= 1.0 and image_shape is known, we might convert.
        # This example just checks if image_shape is not None:
        if image_shape and all(c <= 1.0 for c in coords):
            # It's probably normalized YOLO coords. Convert them:
            h_img, w_img = image_shape
            x_center = coords[0] * w_img
            y_center = coords[1] * h_img
            w_bbox = coords[2] * w_img
            h_bbox = coords[3] * h_img
            x1 = int(x_center - w_bbox / 2)
            y1 = int(y_center - h_bbox / 2)
            x2 = int(x_center + w_bbox / 2)
            y2 = int(y_center + h_bbox / 2)
        else:
            # Assume they're already pixel-based (x1, y1, x2, y2) or (x_center, y_center, w, h).
            # You might add a more robust logic to check which is correct.
            # For a simple approach, let's guess it's top-left and bottom-right.
            # If it's YOLO style in absolute pixels, we do a quick transform:
            # This example uses a naive heuristic:
            x_center, y_center, w_bbox, h_bbox = coords
            # Check if x_center < w_bbox => might be top-left coords
            # Or just assume it's x_center-based if it doesn't make sense as corners
            if x_center < w_bbox or y_center < h_bbox:
                # Probably top-left, bottom-right already
                x1, y1, x2, y2 = map(int, coords)
            else:
                # Convert from x_center-based absolute pixels
                x1 = int(x_center - w_bbox / 2)
                y1 = int(y_center - h_bbox / 2)
                x2 = int(x_center + w_bbox / 2)
                y2 = int(y_center + h_bbox / 2)
    else:
        raise ValueError(f"Unknown bbox format for: {bbox}")

    # Ensure x1 < x2 and y1 < y2
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    return x1, y1, x2, y2

def draw_scan_zone(image, zone, color=(255, 0, 0), thickness=2):
    """
    Draw a rectangular 'scan zone' on the image for demonstration.
    :param image: The image/frame (numpy array).
    :param zone: (x1, y1, x2, y2) for the rectangular zone.
    :param color: The color used for the rectangle.
    :param thickness: The thickness of the rectangle border.
    :return: The annotated image.
    """
    x1, y1, x2, y2 = zone
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, "Scan Zone", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def put_text(image, text, org=(10, 30), color=(0, 255, 0), font_scale=0.7, thickness=2):
    """
    Helper to place text on an image at a certain position.
    :param image: The image/frame.
    :param text: The text to draw.
    :param org: (x, y) position of the text's bottom-left corner.
    :param color: Color for the text (B, G, R).
    :param font_scale: Font scale factor.
    :param thickness: Thickness of the text stroke.
    :return: Annotated image.
    """
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return image

if __name__ == "__main__":
    # Simple demonstration
    test_img_path = "../data/image/sample.jpg"
    # Load sample image
    img = cv2.imread(test_img_path)
    if img is None:
        print("[WARNING] Could not load sample image for demonstration.")
    else:
        # Suppose we have bounding boxes in pixel coords: (class_id, x1, y1, x2, y2)
        sample_bboxes = [
            (0, 100, 100, 200, 200),
            (1, 250, 120, 300, 180),
        ]
        class_list = ["Apple", "Banana"]
        annotated = draw_bboxes(img, sample_bboxes, class_list, color=(0, 255, 0))
        annotated = put_text(annotated, "Sample Overlay Text", org=(50, 50))

        zone = (320, 80, 420, 220)
        annotated = draw_scan_zone(annotated, zone)

        cv2.imshow("draw_utils Demo", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()