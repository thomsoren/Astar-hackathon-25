import os
import json
import random
import shutil
from pathlib import Path

# Hardcoded configuration
SOURCE_DIR = "/home/nicolai/Desktop/NTNU/Semester_2/Hackathon/images/NGD_HACK/"  # Input directory with product folders
TARGET_DIR = "src/data/"  # Output directory (no trailing slash)
PLU_MAPPING_FILE = "/home/nicolai/Desktop/NTNU/Semester_2/Hackathon/Astar-hackathon-25/DELIVERY/PROBLEM1/better_delivery/src/plu_mapping.json"  # PLU mapping file

def load_plu_mapping(mapping_file):
    """Les inn PLU mapping fra JSON-fil."""
    with open(mapping_file, 'r') as f:
        return json.load(f)

def create_directory_structure(target_dir):
    """Opprett YOLO-datastruktur, men kun med 'test'-mapper."""
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    (Path(target_dir) / "images/test").mkdir(parents=True, exist_ok=True)
    (Path(target_dir) / "labels/test").mkdir(parents=True, exist_ok=True)

def convert_annotation(annotation_path, plu_to_index):
    """
    Konverter annotasjon til YOLO-format: 
    [class x_center y_center width height].
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    
    label_data = data['label'][0]
    plu = label_data['label']
    class_idx = plu_to_index[plu]
    
    topX, topY = float(label_data['topX']), float(label_data['topY'])
    bottomX, bottomY = float(label_data['bottomX']), float(label_data['bottomY'])
    
    x_center = (topX + bottomX) / 2
    y_center = (topY + bottomY) / 2
    width = bottomX - topX
    height = bottomY - topY
    
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_dataset(source_dir, target_dir, plu_mapping, plu_to_index):
    """
    Gå rekursivt gjennom source_dir og kopier bilder/annotasjoner til test-mappene.
    PLU hentes fra mappen der bildet ligger. Returnerer antall flyttede bilder.
    """
    count = 0
    source_path = Path(source_dir)
    
    for img_file in source_path.rglob('*.png'):
        # Hopp over filer som ender med "_bb.png"
        if '_bb.png' in img_file.name:
            continue
        
        # Bruk navnet på overliggende mappe som PLU
        product_folder = img_file.parent.name
        if product_folder not in plu_mapping:
            continue
        
        annotation_file = img_file.parent / f"{img_file.stem}.txt"
        if annotation_file.exists():
            dest_img = Path(target_dir) / 'images' / 'test' / img_file.name
            shutil.copy(img_file, dest_img)
            
            yolo_ann = convert_annotation(annotation_file, plu_to_index)
            dest_ann = Path(target_dir) / 'labels' / 'test' / f"{img_file.stem}.txt"
            with open(dest_ann, 'w') as f:
                f.write(yolo_ann + '\n')
            
            count += 1

    return count

def create_yaml_config(target_dir, plu_mapping):
    """
    Lag YOLO dataset-konfigurasjonsfil. 
    Her spesifiserer vi kun 'test' siden alt havner i test.
    """
    yaml_content = (
        "path: ../\n"
        "test: images/test\n"
        f"nc: {len(plu_mapping)}\n"
        "names:\n"
    )
    for idx, (plu, name) in enumerate(plu_mapping.items()):
        yaml_content += f"  {idx}: '{name}'\n"
    
    with open(f"{target_dir}/dataset.yaml", 'w') as f:
        f.write(yaml_content)

def main():
    plu_mapping = load_plu_mapping(PLU_MAPPING_FILE)
    plu_to_index = {plu: idx for idx, plu in enumerate(plu_mapping.keys())}

    create_directory_structure(TARGET_DIR)
    moved_count = process_dataset(SOURCE_DIR, TARGET_DIR, plu_mapping, plu_to_index)
    create_yaml_config(TARGET_DIR, plu_mapping)

    print(f"\nYOLO-datasett er opprettet under '{TARGET_DIR}'.")
    print(f"Antall klasser: {len(plu_mapping)}.")
    print(f"Antall bilder forsøkt flyttet: {moved_count}")

if __name__ == "__main__":
    main()
