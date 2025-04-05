#!/usr/bin/env python3
"""
track_objects.py

Demonstrates multi-object tracking for video streams using YOLOv8's built-in tracker (ByteTrack).
Alternatively, you can integrate an external tracker like BoT-SORT or Norfair.
"""

import os
import sys
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def track_video_with_yolo(model_weights, source, output_dir, tracker_config='bytetrack.yaml', show=True, save=True):
    """
    Tracks objects in a video source using YOLOv8's built-in .track() method with ByteTrack by default.

    :param model_weights: Path to the YOLO weights file (e.g., ../models/yolo/best.pt).
    :param source: Path to a video file or webcam index (e.g. 0).
    :param output_dir: Directory to save the tracked output video or frames.
    :param tracker_config: Name/path of the tracker config (e.g., bytetrack.yaml).
    :param show: If True, display the tracking feed in a window.
    :param save: If True, save the output as a video file or frames in output_dir.
    """
    if not YOLO:
        print("[ERROR] 'ultralytics' not installed. Please install via `pip install ultralytics`.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Loading YOLO model from: {model_weights}")
    model = YOLO(model_weights)

    # The track() method from Ultralytics handles:
    #   - reading the video frames
    #   - running detection
    #   - associating detections with track IDs (ByteTrack)
    #   - optionally saving the output
    # You can pass show=True, save=True directly to .track(), or handle them manually.

    model.track(
        source=source,
        conf=0.5,               # Confidence threshold for detections
        iou=0.5,                # IoU threshold for NMS
        tracker=tracker_config, # The tracker config file (ByteTrack is default)
        show=show,              # Show live tracking feed
        save=save,              # Save output video to "runs/track/expX/" by default
        project=output_dir,     # Where to save results if save=True
        name='tracked_results'  # Subfolder for the results
    )

def main():
    """
    A simple command-line entry point for video-based object tracking with YOLOv8.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Track objects in a video using YOLOv8 + ByteTrack.")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to your trained YOLO model weights (e.g. best.pt).')
    parser.add_argument('--source', type=str, default='0',
                        help='Path to video file or webcam index (e.g. 0 for default webcam).')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save tracked video/frames.')
    parser.add_argument('--tracker_config', type=str, default='bytetrack.yaml',
                        help='Path or name of the tracker configuration (default=bytetrack.yaml).')
    parser.add_argument('--show', action='store_true',
                        help='Display the live tracking output in a window.')
    parser.add_argument('--nosave', action='store_true',
                        help='Disable saving the output video.')
    
    args = parser.parse_args()
    save_flag = not args.nosave
    
    track_video_with_yolo(
        model_weights=args.weights,
        source=args.source,
        output_dir=args.output_dir,
        tracker_config=args.tracker_config,
        show=args.show,
        save=save_flag
    )

if __name__ == "__main__":
    main()