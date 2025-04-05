import os
import cv2
import random
import albumentations as A
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Original train folders
IMG_DIR = os.path.join(SCRIPT_DIR, "data", "images", "train")
LABEL_DIR = os.path.join(SCRIPT_DIR, "data", "labels", "train")

# Where augmented images/labels will be saved (here, same as original)
AUG_IMG_DIR = IMG_DIR
AUG_LABEL_DIR = LABEL_DIR

os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)

# Albumentations pipeline
augmentor = A.Compose(
    [
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.3),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.Affine(scale=(0.9, 1.1), rotate=(-5, 5), p=0.2),
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.1,
        check_each_transform=False  # don't fail mid-transform if out of [0,1]
    )
)

def read_yolo_labels(label_path):
    """Reads YOLO label file -> (bboxes, class_labels)."""
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])
                bboxes.append([xc, yc, w, h])
                class_labels.append(cls_id)
    return bboxes, class_labels

def save_yolo_labels(save_path, bboxes, class_labels):
    """Saves YOLO bboxes + class labels to .txt."""
    with open(save_path, 'w') as f:
        for cls_id, (xc, yc, w, h) in zip(class_labels, bboxes):
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def clamp_or_skip_bboxes(bboxes, class_labels):
    """
    Ensures all boxes are strictly within [0,1].
    If a box is out of range, clamp or discard it.
    """
    valid_bboxes = []
    valid_labels = []
    for (xc, yc, w, h), lbl in zip(bboxes, class_labels):
        left   = xc - w/2
        right  = xc + w/2
        top    = yc - h/2
        bottom = yc + h/2
        
        # Skip if box is completely out of image
        if right < 0 or left > 1 or bottom < 0 or top > 1:
            continue
        
        # Clamp to [0,1]
        left   = max(0.0, min(1.0, left))
        right  = max(0.0, min(1.0, right))
        top    = max(0.0, min(1.0, top))
        bottom = max(0.0, min(1.0, bottom))
        
        # Recompute YOLO
        new_xc = (left + right) / 2
        new_yc = (top + bottom) / 2
        new_w  = right - left
        new_h  = bottom - top
        
        # Optionally skip if new_w or new_h < a threshold...
        valid_bboxes.append([new_xc, new_yc, new_w, new_h])
        valid_labels.append(lbl)
    return valid_bboxes, valid_labels

def gather_images_by_plu():
    """
    Reads all images in IMG_DIR, groups them by PLU prefix in the filename
    (everything before the first dash).
    Example: "7038010013966-320.png" => category "7038010013966".
    """
    cat_to_items = defaultdict(list)
    image_files = [
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    for img_file in image_files:
        # Extract PLU from the filename
        # e.g. "7038010013966-320.png" => "7038010013966-320" => split("-") => "7038010013966"
        base_name = os.path.splitext(img_file)[0]  # "7038010013966-320"
        parts = base_name.split("-", maxsplit=1)
        if len(parts) == 1:
            # If no dash found, fallback to entire filename
            plu = parts[0]
        else:
            plu = parts[0]  # the text before the dash
        
        # Full paths
        img_path = os.path.join(IMG_DIR, img_file)
        label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
        
        cat_to_items[plu].append((img_path, label_path))
    
    return cat_to_items

def augment_needed_images(cat_to_items, target_count=150):
    """
    For each PLU (category), if the number of images < target_count,
    produce enough augmented images to reach target_count.
    Randomly picks base images from that PLU's existing set.
    """
    for plu, items in cat_to_items.items():
        current_count = len(items)
        if current_count >= target_count:
            # Already meets or exceeds 150
            continue
        
        needed = target_count - current_count
        print(f"PLU {plu}: has {current_count}, needs {needed} more images.")
        
        for i in range(needed):
            # Pick a random base image/label from this category
            base_img_path, base_label_path = random.choice(items)
            
            image = cv2.imread(base_img_path)
            if image is None:
                continue
            
            bboxes, class_labels = read_yolo_labels(base_label_path)
            bboxes, class_labels = clamp_or_skip_bboxes(bboxes, class_labels)
            if not bboxes:
                continue
            
            # Apply Albumentations
            transformed = augmentor(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_labels = transformed['class_labels']
            
            # Make a new filename
            base_name = Path(base_img_path).stem  # e.g. "7038010013966-320"
            aug_img_filename = f"{base_name}_aug_{plu}_{i}.jpg"
            aug_label_filename = f"{base_name}_aug_{plu}_{i}.txt"
            
            aug_image_path = os.path.join(AUG_IMG_DIR, aug_img_filename)
            aug_label_path = os.path.join(AUG_LABEL_DIR, aug_label_filename)
            
            # Save to disk
            cv2.imwrite(aug_image_path, aug_image)
            save_yolo_labels(aug_label_path, aug_bboxes, aug_labels)
            
            # Optionally add newly created items to the list
            cat_to_items[plu].append((aug_image_path, aug_label_path))
            
            print(f"  -> Augmented {aug_image_path}")

def main():
    # 1. Gather images by PLU prefix
    cat_to_items = gather_images_by_plu()
    
    # 2. Augment up to 150 images per PLU
    augment_needed_images(cat_to_items, target_count=150)

if __name__ == "__main__":
    main()
