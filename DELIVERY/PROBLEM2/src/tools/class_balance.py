#!/usr/bin/env python3
"""
Class balance checker for YOLO training pipeline
"""

import os
import yaml
import json
from collections import defaultdict
from pathlib import Path
import cv2

def validate_paths(data_yaml):
    """Validate all dataset paths exist"""
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    
    missing = []
    for split in ['train', 'val', 'test']:
        path = Path(data[split])
        if not path.exists():
            missing.append(str(path))
    
    return not bool(missing), missing

def count_class_instances(data_yaml, min_instances=50):
    """Count images and annotations per class"""
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    
    counts = defaultdict(int)
    class_names = data['names']
    
    for split in ['train', 'val']:
        img_dir = Path(data[split])
        label_dir = img_dir.parent / 'labels'
        
        for img_path in img_dir.glob('*.jpg'):
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        class_id = int(line.split()[0])
                        counts[class_id] += 1
    
    report = {
        'status': 'balanced',
        'classes': [],
        'missing': [],
        'below_threshold': []
    }
    
    for class_id, name in enumerate(class_names):
        count = counts.get(class_id, 0)
        status = 'ok' if count >= min_instances else 'warning'
        
        if count == 0:
            report['missing'].append(class_id)
        elif count < min_instances:
            report['below_threshold'].append(class_id)
        
        report['classes'].append({
            'id': class_id,
            'name': name,
            'count': count,
            'status': status
        })
    
    if report['missing'] or report['below_threshold']:
        report['status'] = 'imbalanced'
    
    return report

def generate_augmentation_strategy(report):
    """Generate augmentation recommendations"""
    strategy = {}
    for cls in report['classes']:
        if cls['status'] == 'warning':
            count = max(1, cls['count'])  # Prevent division by zero
            multiplier = max(2, 50 // count)
            strategy[cls['id']] = {
                'multiplier': multiplier,
                'techniques': [
                    'flip', 
                    'rotate', 
                    'brightness',
                    'synthetic' if count < 10 else None
                ]
            }
    return strategy

def check_balance(data_yaml, min_instances=50):
    """Main balance check entry point"""
    paths_valid, missing = validate_paths(data_yaml)
    if not paths_valid:
        return {
            'status': 'error',
            'message': f"Missing dataset paths: {missing}"
        }
    
    report = count_class_instances(data_yaml, min_instances)
    if report['status'] == 'imbalanced':
        report['augmentation'] = generate_augmentation_strategy(report)
    
    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='dataset.yaml path')
    parser.add_argument('--min', type=int, default=50, help='minimum instances')
    args = parser.parse_args()
    
    result = check_balance(args.data, args.min)
    print(json.dumps(result, indent=2))
