import argparse
import json
import shutil
import random
import os
from pathlib import Path

# For evaluation and plotting
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --------------------
# 1. Parse Args
# --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Convert validation data into YOLO test dataset and evaluate.")
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to the validation folder (same structure as training data).')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Where to create the YOLO-style /images/test and /labels/test directories.')
    parser.add_argument('--plu_mapping_file', type=str, default='src/plu_mapping.json',
                        help='Path to the PLU mapping JSON file.')
    return parser.parse_args()

# --------------------
# 2. Helper Functions
# --------------------
def load_plu_mapping(mapping_file):
    """Load PLU mapping from JSON file."""
    with open(mapping_file, 'r') as f:
        return json.load(f)

def create_directory_structure_for_test(target_dir):
    """
    Create YOLO directory structure for test only:
      target_dir/images/test
      target_dir/labels/test
    """
    Path(target_dir, 'images', 'test').mkdir(parents=True, exist_ok=True)
    Path(target_dir, 'labels', 'test').mkdir(parents=True, exist_ok=True)

def convert_annotation(annotation_path, plu_to_index):
    """Convert bounding box from custom JSON format to YOLO (class, x_center, y_center, width, height)."""
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    label_data = data['label'][0]
    plu = label_data['label']
    class_idx = plu_to_index[plu]
    
    # Original bounding box coords
    topX, topY = float(label_data['topX']), float(label_data['topY'])
    bottomX, bottomY = float(label_data['bottomX']), float(label_data['bottomY'])
    
    # Convert to YOLO [class x_center y_center width height]
    x_center = (topX + bottomX) / 2.0
    y_center = (topY + bottomY) / 2.0
    width = bottomX - topX
    height = bottomY - topY
    
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_all_as_test(source_dir, target_dir, plu_mapping, plu_to_index):
    """
    Scan the given source_dir (with subfolders for each PLU),
    gather images and annotations, and copy them into:
        target_dir/images/test
        target_dir/labels/test
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Collect all (image, annotation) pairs
    image_annotations = []
    
    for plu in plu_mapping:
        product_dir = source_dir / plu
        if not product_dir.exists():
            # Skip if this PLU folder doesn't exist in validation set
            continue
        
        for img_file in product_dir.glob('*.png'):
            # Skip the _bb.png images
            if '_bb.png' in img_file.name:
                continue
            
            # Corresponding .txt for bounding box
            ann_file = product_dir / f"{img_file.stem}.txt"
            if ann_file.exists():
                image_annotations.append((img_file, ann_file))
    
    # Process all as "test" data
    for img_path, ann_path in image_annotations:
        # Copy the image over
        dest_img = target_dir / 'images' / 'test' / img_path.name
        shutil.copy(img_path, dest_img)
        
        # Convert and save YOLO annotation
        yolo_line = convert_annotation(ann_path, plu_to_index)
        dest_label = target_dir / 'labels' / 'test' / f"{img_path.stem}.txt"
        with open(dest_label, 'w') as f:
            f.write(yolo_line + '\n')

def create_yaml_config_for_test(target_dir, plu_mapping):
    """
    Creates a dataset.yaml file that references only the 'test' split,
    using absolute paths for 'path' and 'test'.
    """
    # Resolve the target directory to an absolute path
    abs_target_dir = Path(target_dir).resolve().as_posix()
    
    nc = len(plu_mapping)  # number of classes
    
    yaml_content = (
        f"path: {abs_target_dir}\n"   # Absolute path to the dataset root
        f"train:\n"                  # Not used, so left empty
        f"val:\n"                    # Not used, so left empty
        f"test: {abs_target_dir}/images/test\n"  # Absolute path for test images
        f"nc: {nc}\n"
        f"names:\n"
    )
    
    # Each index -> class name
    for idx, (plu, name) in enumerate(plu_mapping.items()):
        yaml_content += f"  {idx}: '{name}'\n"
    
    # Write out the dataset.yaml
    yaml_path = Path(target_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

# --------------------
# 3. Main Function
# --------------------
def main():
    args = parse_args()
    
    # Step A: Convert incoming data to YOLO test set
    plu_mapping = load_plu_mapping(args.plu_mapping_file)
    plu_to_index = {plu: idx for idx, plu in enumerate(plu_mapping.keys())}

    # 1. Create YOLO test directories
    create_directory_structure_for_test(args.output_dir)

    # 2. Process the entire val_dir -> test set
    process_all_as_test(args.val_dir, args.output_dir, plu_mapping, plu_to_index)

    # 3. Create dataset.yaml
    create_yaml_config_for_test(args.output_dir, plu_mapping)

    print(f"\nAll data from '{args.val_dir}' was converted to YOLO 'test' format in '{args.output_dir}'.")
    print("Contents:")
    print(f"  - {args.output_dir}/images/test")
    print(f"  - {args.output_dir}/labels/test")
    print(f"  - {args.output_dir}/dataset.yaml")

    # Step B: Evaluate with YOLO model and visualize results
    #
    # If your model weights are in `model/` folder as `best.pt`, update accordingly:
    MODEL_WEIGHTS = "model/best.pt"
    
    # 4. Load the YOLO model
    model = YOLO(MODEL_WEIGHTS)

    # 5. Evaluate on the test set
    dataset_yaml = os.path.join(args.output_dir, 'dataset.yaml')
    results = model.val(data=dataset_yaml, split='test', conf=0.25)

    # 6. Print out summary metrics
    #    (depending on YOLO version, you can access metrics differently)
    print("\n=== Evaluation Metrics ===")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP75:    {results.box.map75:.4f}")
    print(f"Per-class mAP50-95: {results.box.maps}")

    # 7. Plot per-class mAP50-95 in a bar chart
    category_maps = results.box.maps  # list of mAP50-95 values for each class
    class_names = list(plu_mapping.values())  # names from the PLU mapping

    # Safety check: make sure lengths match
    if len(category_maps) != len(class_names):
        print("Warning: mismatch between classes in mapping and result metrics.")
    
    # Create a bar plot
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, category_maps)
    plt.title("Per-Class mAP50-95")
    plt.xlabel("Classes")
    plt.ylabel("mAP50-95")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save and/or show the plot
    plot_path = "accuracy_plot.png"
    plt.savefig(plot_path)
    plt.show()

    print(f"\nPer-class accuracy plot saved as: {plot_path}")

if __name__ == "__main__":
    main()
