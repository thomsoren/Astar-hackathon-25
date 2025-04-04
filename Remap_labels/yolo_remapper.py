import os
import json
import glob

# Finn katalogen der scriptet ligger
script_dir = os.path.dirname(os.path.abspath(__file__))

# Anta at "Data" ligger i samme overordnede mappe som "Private_files"
data_dir = os.path.join(os.path.dirname(script_dir), "Data")

# Mapping: original mappenavn (kategori) -> ny numerisk label
label_mapping = {}
current_label = 0

# Gå gjennom hver undermappe i data_dir
for folder in sorted(os.listdir(data_dir)):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        label_mapping[folder] = current_label
        print(f"Folder '{folder}' gets label {current_label}")
        current_label += 1

        # Finn alle .txt-filer i denne mappen
        for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            yolo_lines = []
            # Konverter hver annotasjon i filen
            for annot in data.get("label", []):
                topX = annot["topX"]
                topY = annot["topY"]
                bottomX = annot["bottomX"]
                bottomY = annot["bottomY"]

                # Beregn senter, bredde og høyde
                x_center = (topX + bottomX) / 2
                y_center = (topY + bottomY) / 2
                width = bottomX - topX
                height = bottomY - topY

                # Bruk ny label basert på mappen
                yolo_lines.append(f"{label_mapping[folder]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Skriv ut data til en kopi med _yolo.txt-suffiks i samme mappe
            base_name = os.path.splitext(file_path)[0]
            output_path = base_name + "_yolo.txt"
            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    for line in yolo_lines:
                        f.write(line + "\n")
                print(f"Converted {file_path} -> {output_path}")
            except Exception as e:
                print(f"Error writing {output_path}: {e}")

# Opprett YAML-fil med dataset-konfigurasjon
yaml_lines = [
    "path: ../data/yolo_dataset",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    f"nc: {len(label_mapping)}",
    "names:"
]

# Bruk original mappenavn (som representerer kategorien) for mapping
for label in sorted(label_mapping.items(), key=lambda x: x[1]):
    # label[0] er mappenavn, label[1] er den nye labelen
    yaml_lines.append(f"  {label[1]}: '{label[0]}'")

# Lag YAML-fil (for eksempel "dataset.yaml" i samme mappe som scriptet)
yaml_file_path = os.path.join(script_dir, "dataset.yaml")
try:
    with open(yaml_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
    print(f"YAML file saved to {yaml_file_path}")
except Exception as e:
    print(f"Error writing YAML file: {e}")
