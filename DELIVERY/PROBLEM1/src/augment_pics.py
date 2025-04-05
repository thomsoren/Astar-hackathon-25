import os
import cv2
import albumentations as A

# 1. Define paths relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "data", "images", "train")
LABEL_DIR = os.path.join(SCRIPT_DIR, "data", "labels", "train")

# 2. Define where augmented files will be saved
AUG_IMG_DIR = os.path.join(SCRIPT_DIR, "data", "images", "train")
AUG_LABEL_DIR = os.path.join(SCRIPT_DIR, "data", "labels", "train")

# Create directories if they don't exist
for dir_path in [IMG_DIR, LABEL_DIR, AUG_IMG_DIR, AUG_LABEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 3. Define your augmentation pipeline
#    (Adjust probabilities and parameters as needed)
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
],
    bbox_params=A.BboxParams(
        format='yolo',        # YOLO format: [class, x_center, y_center, w, h]
        label_fields=['class_labels'],
        min_area=0,           # remove bboxes smaller than this area (in pixels)
        min_visibility=0.1,   # remove bboxes if < this fraction of area remains
    )
)

def read_yolo_labels(label_path):
    """
    Reads YOLO label file and returns lists of [x_center, y_center, w, h]
    and list of class IDs.
    """
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            class_id, x_c, y_c, w, h = line.split()
            class_labels.append(int(class_id))
            bboxes.append([float(x_c), float(y_c), float(w), float(h)])
    return bboxes, class_labels

def save_yolo_labels(save_path, bboxes, class_labels):
    """
    Saves bounding boxes (YOLO format) and class IDs to a .txt file.
    bboxes is a list of [x_center, y_center, w, h].
    class_labels is a list of class IDs.
    """
    with open(save_path, 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            x_c, y_c, w, h = bbox
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def augment_dataset(num_aug=5):
    """
    Creates 'num_aug' augmented images per original image,
    writing them to AUG_IMG_DIR and AUG_LABEL_DIR.
    """
    # Get all image file names
    all_images = [
        f for f in os.listdir(IMG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    
    for img_file in all_images:
        # Original image path and label path
        img_path = os.path.join(IMG_DIR, img_file)
        label_name = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_name)
        
        if not os.path.exists(label_path):
            # Skip if there's no matching label
            print(f"No label found for {img_file}, skipping.")
            continue
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to read {img_file}, skipping.")
            continue
        
        height, width, _ = image.shape
        
        # Read bounding boxes in YOLO format
        bboxes, class_labels = read_yolo_labels(label_path)
        
        for aug_i in range(num_aug):
            # Apply augmentation
            transformed = augmentor(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_class_labels = transformed['class_labels']
            
            # Construct new filenames
            # e.g. "image1_aug_0.jpg", "image1_aug_0.txt"
            base_name = os.path.splitext(img_file)[0]
            aug_img_filename = f"{base_name}_aug_{aug_i}.jpg"
            aug_label_filename = f"{base_name}_aug_{aug_i}.txt"
            
            # Save augmented image
            aug_image_path = os.path.join(AUG_IMG_DIR, aug_img_filename)
            cv2.imwrite(aug_image_path, aug_image)
            
            # Save augmented labels
            aug_label_path = os.path.join(AUG_LABEL_DIR, aug_label_filename)
            # Convert bounding boxes (still in YOLO format) back to file
            save_yolo_labels(aug_label_path, aug_bboxes, aug_class_labels)

            print(f"Saved {aug_image_path} & {aug_label_path}")


if __name__ == "__main__":
    # Create 5 augmented images per original
    augment_dataset(num_aug=2)
