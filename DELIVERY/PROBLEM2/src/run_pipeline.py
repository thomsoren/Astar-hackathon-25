#!/usr/bin/env python3
"""
run_pipeline.py

A single script to orchestrate:
1) Frame extraction (optional)
2) Annotation (optional)
3) YOLO inference (optional)
4) Object tracking (optional)

Usage example:
  python run_pipeline.py \
    --video ../data/video/myvideo.mp4 \
    --extract \
    --annotate \
    --infer \
    --track \
    --weights ../models/yolo/best.pt \
    --annotation_dir ../data/annotations \
    --conf 0.5
"""

import argparse
import os
import sys
import json
import time

# Import our local modules
import video_processing
import annotation_tool
import yolo_inference
import track_objects

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an end-to-end pipeline for YOLO detection and tracking."
    )
    # --- General / Shared ---
    parser.add_argument('--weights', type=str, default='../models/yolo/best.pt',
                        help='Path to YOLO weights file.')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detection.')
    parser.add_argument('--video', type=str, default=None,
                        help='Video file path for extraction or inference.')
    parser.add_argument('--annotation_dir', type=str, default='../data/annotations',
                        help='Directory to store or load annotations.')
    
    # --- Steps toggles ---
    parser.add_argument('--extract', action='store_true',
                        help='Extract frames from the specified video.')
    parser.add_argument('--annotate', action='store_true',
                        help='Run annotation tool on a folder of images.')
    parser.add_argument('--infer', action='store_true',
                        help='Run YOLO inference on images or video.')
    parser.add_argument('--track', action='store_true',
                        help='Run YOLO multi-object tracking on the specified video.')
    
    # --- Extraction options ---
    parser.add_argument('--extract_skip', type=int, default=0,
                        help='Number of frames to skip when extracting. Default=0 (no skip).')
    parser.add_argument('--extract_limit', type=int, default=None,
                        help='Max number of frames to extract from video.')
    parser.add_argument('--frame_output_dir', type=str, default='../data/extracted_frames',
                        help='Where to save extracted frames if --extract is used.')
    
    # --- Annotation options ---
    parser.add_argument('--img_dir', type=str, default='../data/extracted_frames',
                        help='Directory of images to annotate.')
    
    # --- Inference options ---
    parser.add_argument('--infer_source', type=str, default=None,
                        help='Source for YOLO inference: single image, folder, or video file. \
                              If not given, defaults to --video or webcam.')
    parser.add_argument('--infer_show', action='store_true', default=True,
                        help='Display YOLO inference output in a window (default: True).')
    parser.add_argument('--infer_save', action='store_true',
                        help='Save YOLO inference output to disk.') 
    parser.add_argument('--no_show', action='store_false', dest='infer_show',
                        help='Disable showing inference output window.')
    parser.add_argument('--results_dir', type=str, default='../results',
                        help='Directory to store inference or tracking results.')
    
    # --- Tracking options ---
    parser.add_argument('--tracker_config', type=str, default='bytetrack.yaml',
                        help='Tracker config for YOLO tracking (ByteTrack by default).')
    parser.add_argument('--track_show', action='store_true',
                        help='Display tracking output in a window.')
    parser.add_argument('--nosave_track', action='store_true',
                        help='Disable saving the tracking output.')
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Check class balance before training
    if args.infer or args.track:
        from tools.class_balance import check_balance
        balance_report = check_balance('DELIVERY/PROBLEM2/data/dataset.yaml')
        if balance_report['status'] == 'error':
            print(f"[ERROR] Dataset validation failed: {balance_report['message']}")
            sys.exit(1)
        elif balance_report['status'] == 'imbalanced':
            print("[WARNING] Class imbalance detected:")
            for cls in balance_report['classes']:
                if cls['status'] == 'warning':
                    print(f"  - {cls['name']}: {cls['count']} instances (min 50 recommended)")

    # Load PLU mapping
    plu_path = os.path.join(os.path.dirname(__file__), '../../../plu_mapping.json')
    with open(plu_path) as f:
        plu_map = json.load(f)

    # If no steps specified, default to infer + track
    if not any([args.extract, args.annotate, args.infer, args.track]):
        print("[INFO] No steps specified - defaulting to inference and tracking")
        args.infer = True
        args.track = True

    # Step 1: Extract frames from video (optional)
    if args.extract:
        if not args.video:
            print("[ERROR] No video provided for frame extraction. Use --video <path>.")
            sys.exit(1)
        print("[INFO] Extracting frames from video ...")
        start_time = time.time()
        frame_count = video_processing.extract_frames_from_video(
            video_path=args.video,
            output_dir=args.frame_output_dir,
            skip=args.extract_skip,
            limit=args.extract_limit
        )
        duration = time.time() - start_time
        print(f"[INFO] Extracted {frame_count} frames in {duration:.2f} seconds\n")

    # Step 2: Annotation (optional)
    if args.annotate:
        print("[INFO] Running annotation tool ...")
        start_time = time.time()
        annotation_count = annotation_tool.annotate_images(
            img_dir=args.img_dir,
            output_dir=args.annotation_dir,
            model_weights=args.weights,
            conf_threshold=args.conf
        )
        duration = time.time() - start_time
        print(f"[INFO] Annotated {annotation_count} images in {duration:.2f} seconds\n")

    # Step 3: YOLO Inference (optional)
    if args.infer:
        infer_source = args.infer_source or args.video or '0'
        print(f"[INFO] Running YOLO inference on source={infer_source} ...")
        start_time = time.time()
        
        # Create an argparse.Namespace object with the parameters
        inference_args = argparse.Namespace(
            source=infer_source,
            weights=args.weights,
            conf=args.conf,
            project=args.results_dir,
            show=args.infer_show,
            save=args.infer_save
        )
        try:
            result = yolo_inference.main(inference_args)
            duration = time.time() - start_time
            if result:
                print(f"[INFO] Processed {result['frame_count']} frames in {duration:.2f} seconds\n")
            else:
                print(f"[INFO] Inference completed in {duration:.2f} seconds but no results returned")
        except Exception as e:
            print(f"[ERROR] Inference failed: {str(e)}")

    # Step 4: Multi-object Tracking (optional)
    if args.track:
        if not args.video:
            print("[ERROR] No video provided for tracking. Use --video <path>.")
            sys.exit(1)
        print("[INFO] Running YOLO tracking ...")
        start_time = time.time()
        save_flag = not args.nosave_track
        track_result = track_objects.track_video_with_yolo(
            model_weights=args.weights,
            source=args.video,
            output_dir=args.results_dir,
            tracker_config=args.tracker_config,
            show=args.track_show,
            save=save_flag,
        )
        duration = time.time() - start_time
        print(f"[INFO] Tracked {track_result['frame_count']} frames in {duration:.2f} seconds\n")

    print("[INFO] Pipeline finished successfully.")

if __name__ == "__main__":
    main()
