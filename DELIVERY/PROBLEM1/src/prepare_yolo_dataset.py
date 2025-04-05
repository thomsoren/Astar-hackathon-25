import os
import json
import random
import shutil
from pathlib import Path

# Hardcoded configuration
SOURCE_DIR = "../data"  # Input directory with product folders
TARGET_DIR = "data/"  # Output directory (no trailing slash)

PLU_MAPPING_FILE = "plu_mapping.json"  # PLU mapping file
SPLIT_RATIOS = {'train': 0.7, 'val': 0.2, 'test': 0.1}  # Train/val/test split

def load_plu_mapping(mapping_file):
    """Load PLU mapping from JSON file"""
    with open(mapping_file, 'r') as f:
        return json.load(f)

def create_directory_structure(target_dir):
    """Create YOLO dataset directory structure"""
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        Path(f"{target_dir}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{target_dir}/labels/{split}").mkdir(parents=True, exist_ok=True)

def convert_annotation(annotation_path, plu_to_index):
    """Convert JSON annotation to YOLO format"""
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    label_data = data['label'][0]
    plu = label_data['label']
    class_idx = plu_to_index[plu]
    
    # Convert coordinates to YOLO format [class x_center y_center width height]
    topX, topY = float(label_data['topX']), float(label_data['topY'])
    bottomX, bottomY = float(label_data['bottomX']), float(label_data['bottomY'])
    
    x_center = (topX + bottomX) / 2
    y_center = (topY + bottomY) / 2
    width = bottomX - topX
    height = bottomY - topY
    
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_dataset(source_dir, target_dir, plu_mapping, plu_to_index, split_ratios):
    """Process all images and annotations, split into train/val/test"""
    image_annotations = []
    
    # Collect all valid image/annotation pairs
    for plu in plu_mapping:
        product_dir = Path(source_dir) / plu
        if not product_dir.exists():
            continue
            
        for img_file in product_dir.glob('*.png'):
            if '_bb.png' in img_file.name:
                continue
                
            annotation_file = product_dir / f"{img_file.stem}.txt"
            if annotation_file.exists():
                image_annotations.append((img_file, annotation_file))
    
    # Shuffle and split
    random.shuffle(image_annotations)
    total = len(image_annotations)
    train_end = int(total * split_ratios['train'])
    val_end = train_end + int(total * split_ratios['val'])
    
    splits = {
        'train': image_annotations[:train_end],
        'val': image_annotations[train_end:val_end],
        'test': image_annotations[val_end:]
    }
    
    # Process each split
    for split_name, items in splits.items():
        for img_path, ann_path in items:
            img_name = img_path.name
            # Copy image
            dest_img = Path(target_dir) / 'images' / split_name / img_name
            shutil.copy(img_path, dest_img)
            # Save converted annotation
            yolo_ann = convert_annotation(ann_path, plu_to_index)
            dest_ann = Path(target_dir) / 'labels' / split_name / f"{img_path.stem}.txt"
            with open(dest_ann, 'w') as f:
                f.write(yolo_ann + '\n')

def create_yaml_config(target_dir, plu_mapping):
    """Create YOLO dataset configuration file"""
    yaml_content = f"""path: ../
train: C:/Users/light/Documents/ntnu/cogito/Astar-hackathon-25/DELIVERY/PROBLEM1/src/images/train
val: C:/Users/light/Documents/ntnu/cogito/Astar-hackathon-25/DELIVERY/PROBLEM1/src/images/val
test: C:/Users/light/Documents/ntnu/cogito/Astar-hackathon-25/DELIVERY/PROBLEM1/src/images/test
nc: {len(plu_mapping)}
names:
"""
    for idx, (plu, name) in enumerate(plu_mapping.items()):
        yaml_content += f"  {idx}: '{name}'\n"
    
    with open(f"{target_dir}/dataset.yaml", 'w') as f:
        f.write(yaml_content)

def main():
    plu_mapping = load_plu_mapping(PLU_MAPPING_FILE)
    plu_to_index = {plu: idx for idx, plu in enumerate(plu_mapping.keys())}

    create_directory_structure(TARGET_DIR)
    process_dataset(SOURCE_DIR, TARGET_DIR, plu_mapping, plu_to_index, SPLIT_RATIOS)
    create_yaml_config(TARGET_DIR, plu_mapping)

    print(f"\nYOLO dataset successfully created at {TARGET_DIR}")
    print(f"Dataset contains {len(plu_mapping)} classes")
    print(f"Split ratios: Train {SPLIT_RATIOS['train']*100:.0f}%, Val {SPLIT_RATIOS['val']*100:.0f}%, Test {SPLIT_RATIOS['test']*100:.0f}%")

if __name__ == "__main__":
    main()
