#!/usr/bin/env python3
"""
yolo_inference.py

A script to run inference on images or videos using a trained YOLO model.
"""

import argparse
import os
import sys
import cv2

# If using ultralytics YOLOv8:
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed. Please install via `pip install ultralytics`.")
    sys.exit(1)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run YOLO inference on image or video.")
    parser.add_argument('--source', type=str, default='0',
                        help='Path to image or video. "0" or "webcam" for webcam. ')
    parser.add_argument('--weights', type=str, default='../models/yolo/best.pt',
                        help='Path to the YOLO model weights (.pt).')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold for detections.')
    parser.add_argument('--project', type=str, default='../results',
                        help='Folder to save output results.')
    parser.add_argument('--show', action='store_true', default=True,
                        help='Show output frames in a window (default: True).')
    parser.add_argument('--no_show', action='store_false', dest='show',
                        help='Disable showing output window.') 
    parser.add_argument('--save', action='store_true',
                        help='Save annotated images/frames to disk.')
    # Uncomment if you want a tracking mode:
    # parser.add_argument('--track', action='store_true', help='Use YOLO tracking mode')
    return parser.parse_args()

def detect_objects(frame, model):
    """
    Performs object detection on a given frame using the YOLO model.
    Converts normalized coordinates to absolute coordinates.
    """
    results = model(frame)
    detections = []
    for box in results[0].boxes:
        x_center, y_center, width, height = box.xywhn[0].tolist()
        # Convert normalized coords to absolute using frame dimensions
        abs_x = int(x_center * frame.shape[1])
        abs_y = int(y_center * frame.shape[0])
        abs_w = int(width * frame.shape[1])
        abs_h = int(height * frame.shape[0])
        detections.append((abs_x, abs_y, abs_w, abs_h, box.conf, box.cls))
    return detections

def main(opt):
    """
    Main function to load the YOLO model and run inference on the specified source.
    """
    # ------------------------------------------------------
    # 1. Load YOLO Model
    # ------------------------------------------------------
    print(f"[INFO] Loading YOLO model from: {opt.weights}")
    # Resolve weights path relative to script location
    weights_path = os.path.join(os.path.dirname(__file__), '../model/yolo/best.pt')
    if not os.path.exists(weights_path):
        print(f"[ERROR] Model weights not found at: {weights_path}")
        sys.exit(1)
    model = YOLO(weights_path)

    # Create the output directory if needed
    if not os.path.exists(opt.project):
        os.makedirs(opt.project)

    # ------------------------------------------------------
    # 2. Handle Source (Image, Video, or Webcam)
    # ------------------------------------------------------
    # Basic checks for what kind of source we have:
    source = opt.source
    if source.lower() in ["0", "webcam"]:
        # Use webcam (device 0)
        source_type = "webcam"
        video_input = cv2.VideoCapture(0)
    elif os.path.isfile(source) or os.path.isfile(os.path.join(os.path.dirname(__file__), source)):
        # It's a file -> check if it's an image or video by extension
        ext = os.path.splitext(source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            source_type = "image"
        else:
            source_type = "video"
        video_input = cv2.VideoCapture(source)  # Will still work for images, but we handle separately
    else:
        # Might be a folder, or an invalid path
        if os.path.isdir(source):
            print(f"[INFO] Source is a directory: {source}\nProcessing all images in that folder.")
            source_type = "image_directory"
        else:
            print(f"[ERROR] Invalid source path: {source}")
            sys.exit(1)
        video_input = None

    # ------------------------------------------------------
    # 3. Single Image or Multiple Images
    # ------------------------------------------------------
    if source_type == "image":
        print("[INFO] Single image mode")
        # Perform inference
        results = model.predict(source, conf=opt.conf)
        # Visualize results
        annotated_image = results[0].plot()  # results[0] is the first image result
        # Show or save if desired
        if opt.show:
            cv2.imshow("YOLO Inference", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if opt.save:
            output_path = os.path.join(opt.project, os.path.basename(source))
            cv2.imwrite(output_path, annotated_image)
            print(f"[INFO] Saved output to {output_path}")

    elif source_type == "image_directory":
        print("[INFO] Image directory mode")
        image_files = [f for f in os.listdir(source) 
                       if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        for img_name in image_files:
            img_path = os.path.join(source, img_name)
            results = model.predict(img_path, conf=opt.conf)
            annotated_image = results[0].plot()
            if opt.show:
                cv2.imshow("YOLO Inference", annotated_image)
                cv2.waitKey(1)  # show next quickly or press any key to advance
            if opt.save:
                out_path = os.path.join(opt.project, img_name)
                cv2.imwrite(out_path, annotated_image)
                print(f"[INFO] Saved {out_path}")
        if opt.show:
            cv2.destroyAllWindows()

    else:
        # --------------------------------------------------
        # 4. Video / Webcam Inference
        # --------------------------------------------------
        print("[INFO] Video/Camera mode")
        if not video_input.isOpened():
            print("[ERROR] Could not open video or webcam.")
            sys.exit(1)

        # Read until the video is completed or user quits
        while True:
            ret, frame = video_input.read()
            if not ret:
                # End of video stream
                print("[INFO] End of stream or can't read frame.")
                break

            # YOLO inference
            # For each frame, run detection. 
            #   - set conf=opt.conf for detection threshold
            results = model.predict(frame, conf=opt.conf)

            # The model returns a list of results, one per image/frame
            annotated_frame = results[0].plot()

            # Display
            if opt.show:
                cv2.imshow("YOLO Inference", annotated_frame)
                cv2.setWindowProperty("YOLO Inference", cv2.WND_PROP_TOPMOST, 1)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

            # Optional: Save frames to disk
            if opt.save:
                # Just as an example, we name frames by time or frame count
                frame_name = f"frame_{int(video_input.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
                out_path = os.path.join(opt.project, frame_name)
                cv2.imwrite(out_path, annotated_frame)

        video_input.release()
        if opt.show:
            cv2.destroyAllWindows()

    print("[INFO] Inference completed.")

if __name__ == "__main__":
    opt = parse_arguments()
    main(opt)
