import os
import cv2
import albumentations as A
from collections import defaultdict

# Egendefinert funksjon som konverterer en YOLO-boks til VOC, klipper den, og konverterer tilbake til YOLO
def clip_yolo_bbox(bbox):
    x_center, y_center, w, h = bbox
    # Konverter til Pascal VOC-format
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

    # Klipp verdiene til intervallet [0, 1]
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    # Konverter tilbake til YOLO-format
    new_x_center = (x_min + x_max) / 2
    new_y_center = (y_min + y_max) / 2
    new_w = x_max - x_min
    new_h = y_max - y_min
    return [new_x_center, new_y_center, new_w, new_h]

def clip_yolo_bboxes(bboxes, **kwargs):
    return [clip_yolo_bbox(bbox) for bbox in bboxes]

# Definer en Lambda-transformasjon som bruker funksjonen over
clip_bbox = A.Lambda(bbox_func=clip_yolo_bboxes, always_apply=True)


# Base directory for data
BASE_DIR = "Astar-hackathon-25/DELIVERY/PROBLEM1/data"
SPLITS = ["train/", "val/", "test/"]
# Felles funksjoner for lesing og skriving av YOLO-labels
def read_yolo_labels(label_path):
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
    with open(save_path, 'w') as f:
        for cls, bbox in zip(class_labels, bboxes):
            x_c, y_c, w, h = bbox
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

def count_prefixes(image_dir):
    counts = defaultdict(int)
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            prefix = f.split('-')[0]
            counts[prefix] += 1
    return counts

def get_original_images_for_prefix(image_dir, prefix):
    originals = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.startswith(prefix + "-") and "_aug_" not in f:
            originals.append(f)
    return originals

# Augmentor for "train" – generell pipeline
train_augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=20, p=0.5),
    clip_bbox
],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.0,
    )
)

def augment_until_minimum(image_dir, label_dir, min_count=200, num_aug_per_round=1):
    counts = count_prefixes(image_dir)
    for prefix, count in counts.items():
        if count >= min_count:
            continue

        needed = min_count - count
        print(f"Prefix {prefix} har {count} bilder, trenger {needed} flere.")
        originals = get_original_images_for_prefix(image_dir, prefix)
        if not originals:
            print(f"Ingen originale bilder funnet for prefix {prefix}.")
            continue

        aug_round = 0
        while needed > 0:
            print(f"Augmenteringsrunde {aug_round} for prefix {prefix}...")
            for img_file in originals:
                if needed <= 0:
                    break

                img_path = os.path.join(image_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                label_name = base_name + ".txt"
                label_path = os.path.join(label_dir, label_name)

                if not os.path.exists(label_path):
                    print(f"Ingen label funnet for {img_file}, hopper over.")
                    continue

                image = cv2.imread(img_path)
                if image is None:
                    print(f"Kunne ikke lese {img_file}, hopper over.")
                    continue

                bboxes, class_labels = read_yolo_labels(label_path)
                for aug_i in range(num_aug_per_round):
                    transformed = train_augmentor(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    aug_image = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_class_labels = transformed['class_labels']

                    aug_img_filename = f"{base_name}_aug_{aug_round}_{aug_i}.jpg"
                    aug_label_filename = f"{base_name}_aug_{aug_round}_{aug_i}.txt"

                    aug_img_path = os.path.join(image_dir, aug_img_filename)
                    cv2.imwrite(aug_img_path, aug_image)
                    aug_label_path = os.path.join(label_dir, aug_label_filename)
                    save_yolo_labels(aug_label_path, aug_bboxes, aug_class_labels)

                    needed -= 1
                    print(f"Lagret {aug_img_path} og {aug_label_path}. Gjenstående for {prefix}: {needed}")
                    if needed <= 0:
                        break
            aug_round += 1

# Egne pipelines for "val" og "test"
flip_augmentor = A.Compose([
    A.HorizontalFlip(p=1.0),
    clip_bbox
],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.0,
    )
)
rotate_augmentor = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=20, p=1.0),
    clip_bbox
],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.0,
    )
)
gamma_augmentor = A.Compose([
    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
    clip_bbox
],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=0,
        min_visibility=0.0,
    )
)

def augment_fixed_copies(image_dir, label_dir):
    originals = [f for f in os.listdir(image_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_aug_" not in f]
    
    for img_file in originals:
        img_path = os.path.join(image_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        label_name = base_name + ".txt"
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            print(f"Ingen label funnet for {img_file}, hopper over.")
            continue

        image = cv2.imread(img_path)
        if image is None:
            print(f"Kunne ikke lese {img_file}, hopper over.")
            continue

        bboxes, class_labels = read_yolo_labels(label_path)

        augmentations = [
            ("flip", flip_augmentor),
            ("rot", rotate_augmentor),
            ("gamma", gamma_augmentor)
        ]

        for aug_name, aug_pipeline in augmentations:
            transformed = aug_pipeline(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_class_labels = transformed['class_labels']

            aug_img_filename = f"{base_name}_aug_{aug_name}.jpg"
            aug_label_filename = f"{base_name}_aug_{aug_name}.txt"

            aug_img_path = os.path.join(image_dir, aug_img_filename)
            cv2.imwrite(aug_img_path, aug_image)
            aug_label_path = os.path.join(label_dir, aug_label_filename)
            save_yolo_labels(aug_label_path, aug_bboxes, aug_class_labels)

            print(f"Lagret {aug_img_path} og {aug_label_path}")

def main():
    for split in SPLITS:
        print(f"Behandler split: {split}")
        img_dir = os.path.join(BASE_DIR, "images", split)
        label_dir = os.path.join(BASE_DIR, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        if split == "train":
            augment_until_minimum(img_dir, label_dir, min_count=200)
        else:
            augment_fixed_copies(img_dir, label_dir)

if __name__ == "__main__":
    main()