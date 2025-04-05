#!/usr/bin/env python3
"""
io_utils.py

Utility functions for reading and writing files (JSON, text, etc.),
managing paths, and other I/O operations.
"""

import os
import json

def ensure_dir(dir_path):
    """
    Creates the directory if it doesn't already exist.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def read_json(json_path):
    """
    Reads and returns the data from a JSON file.
    Returns None if file does not exist or fails to load.
    """
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load JSON from {json_path}: {e}")
        return None

def write_json(json_path, data, indent=2):
    """
    Writes data to a JSON file. Overwrites if file exists.
    """
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write JSON to {json_path}: {e}")
        return False

def list_files_in_dir(dir_path, extensions=None, recursive=False):
    """
    Lists files in a directory, optionally filtered by extension(s).
    :param dir_path: Path to the directory.
    :param extensions: A tuple/list of valid extensions (e.g. ('.jpg', '.png')).
    :param recursive: If True, search recursively in subfolders.
    :return: List of file paths.
    """
    if not os.path.isdir(dir_path):
        print(f"[WARNING] Directory does not exist: {dir_path}")
        return []
    
    matched_files = []
    if recursive:
        for root, _, files in os.walk(dir_path):
            for f in files:
                if extensions:
                    if os.path.splitext(f)[1].lower() in extensions:
                        matched_files.append(os.path.join(root, f))
                else:
                    matched_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(dir_path):
            full_path = os.path.join(dir_path, f)
            if os.path.isfile(full_path):
                if extensions:
                    if os.path.splitext(f)[1].lower() in extensions:
                        matched_files.append(full_path)
                else:
                    matched_files.append(full_path)
    
    return matched_files

def read_text_file(txt_path):
    """
    Reads a plain text file and returns a list of lines.
    Returns None if the file does not exist or fails to open.
    """
    if not os.path.isfile(txt_path):
        return None
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
        return lines
    except Exception as e:
        print(f"[ERROR] Failed to read text file from {txt_path}: {e}")
        return None

def write_text_file(txt_path, lines):
    """
    Writes a list of lines to a text file, overwriting if it already exists.
    """
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(f"{line}\n")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write text file to {txt_path}: {e}")
        return False

if __name__ == "__main__":
    # Simple demonstration of a few functions:
    test_dir = "../data/test_dir"
    ensure_dir(test_dir)
    print(f"Created or verified directory: {test_dir}")

    json_path = os.path.join(test_dir, "example.json")
    example_data = {"message": "Hello, world!", "count": 123}
    write_json(json_path, example_data)

    loaded_data = read_json(json_path)
    print("[INFO] Loaded JSON data:", loaded_data)