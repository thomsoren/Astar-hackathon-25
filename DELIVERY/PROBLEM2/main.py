import cv2
import os
import json
import math
import csv # Added for CSV output
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse

# --- Configuration ---
MODEL_PATH = 'model/best.pt' # Relative path from main.py
VIDEO_DIR = '../../data/videos'           # Relative path from main.py
OUTPUT_DIR = 'output'                     # Relative path from main.py
MAX_DISTANCE_THRESHOLD = 150               # Max pixel distance to match a detection to a track
MAX_FRAMES_UNSEEN = 20                    # Max frames to keep a track alive without detection
CONFIDENCE_THRESHOLD = 0.6                # Minimum confidence score for YOLO detections
VISUALIZE = True                          # Set to True to show video playback with tracking

# --- Helper Functions ---
def calculate_center(bbox):
    """Calculates the center (x, y) of a bounding box."""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# --- Tracking Logic ---
class Tracker:
    def __init__(self, max_distance, max_unseen):
        # Store track_id -> {coords: list, last_pos: tuple, last_bbox: tuple, unseen_frames: int, active: bool}
        self.tracks = {}
        self.next_track_id = 0
        self.max_distance = max_distance
        self.max_unseen = max_unseen
        # Generate distinct colors for tracks
        np.random.seed(42) # for reproducibility
        self.track_colors = {i: tuple(np.random.randint(0, 255, 3).tolist()) for i in range(100)} # Pre-generate colors for 100 tracks

    def _get_active_tracks(self):
        return {tid: track for tid, track in self.tracks.items() if track['active']}

    def _get_color(self, track_id):
        """Gets a color for a track ID, generating a new one if needed."""
        if track_id not in self.track_colors:
             self.track_colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.track_colors[track_id]

    def update(self, detections_with_boxes, frame_number):
        """
        Updates tracks based on new detections.
        detections_with_boxes: list of tuples [(center_x, center_y), (x1, y1, x2, y2)]
        """
        active_tracks = self._get_active_tracks()
        matched_track_ids = set()
        used_detection_indices = set()

        # Try to match detections to existing active tracks
        if detections_with_boxes and active_tracks:
            # Build cost matrix (distances between centers)
            cost_matrix = np.full((len(active_tracks), len(detections_with_boxes)), float('inf'))
            track_ids = list(active_tracks.keys())
            detection_centers = [det[0] for det in detections_with_boxes] # Extract centers

            for i, track_id in enumerate(track_ids):
                track_pos = active_tracks[track_id]['last_pos']
                for j, det_center in enumerate(detection_centers):
                    dist = euclidean_distance(track_pos, det_center)
                    if dist < self.max_distance:
                        cost_matrix[i, j] = dist

            # Simple greedy matching (could be improved with Hungarian algorithm)
            # Sort potential matches by distance
            matches = []
            rows, cols = np.where(cost_matrix != float('inf'))
            for r, c in zip(rows, cols):
                 matches.append((cost_matrix[r, c], track_ids[r], c)) # (distance, track_id, detection_index)
            
            matches.sort() # Sort by distance (ascending)

            for dist, track_id, det_idx in matches:
                if track_id not in matched_track_ids and det_idx not in used_detection_indices:
                    # Match found
                    det_center, det_bbox = detections_with_boxes[det_idx]
                    self.tracks[track_id]['coords'].append([frame_number, det_center[0], det_center[1]])
                    self.tracks[track_id]['last_pos'] = det_center
                    self.tracks[track_id]['last_bbox'] = det_bbox # Store the bbox
                    self.tracks[track_id]['unseen_frames'] = 0
                    matched_track_ids.add(track_id)
                    used_detection_indices.add(det_idx)


        # Handle unmatched tracks (increment unseen count or deactivate)
        for track_id, track in active_tracks.items():
            if track_id not in matched_track_ids:
                track['unseen_frames'] += 1
                if track['unseen_frames'] > self.max_unseen:
                    track['active'] = False # Deactivate track
                    track['last_bbox'] = None # Clear bbox when inactive

        # Handle unmatched detections (create new tracks)
        for i, (det_center, det_bbox) in enumerate(detections_with_boxes):
            if i not in used_detection_indices:
                new_id = self.next_track_id
                self.tracks[new_id] = {
                    'coords': [[frame_number, det_center[0], det_center[1]]],
                    'last_pos': det_center,
                    'last_bbox': det_bbox, # Store initial bbox
                    'unseen_frames': 0,
                    'active': True
                }
                self.next_track_id += 1

    def get_results(self):
        """Returns the coordinate history for all tracks (active and inactive)."""
        # Filter out tracks that might have been created but never updated significantly (optional)
        # return {tid: track['coords'] for tid, track in self.tracks.items() if len(track['coords']) > 1}
        return {tid: track['coords'] for tid, track in self.tracks.items()}


# --- Main Processing Function ---
def process_video(video_path, model):
    """
    Processes a single video file for object tracking and identifies items near the center line.
    Returns a list of (timestamp, item_label) tuples for items near the center.
    """
    print(f"Processing video: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return [] # Return empty list on error

    tracker = Tracker(max_distance=MAX_DISTANCE_THRESHOLD, max_unseen=MAX_FRAMES_UNSEEN)
    frame_number = 0
    center_items_receipt = [] # List to store (timestamp, label) for this video

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Not used for logic, but good to have
    center_x_frame = frame_width / 2
    center_proximity_threshold = MAX_DISTANCE_THRESHOLD # Use the same threshold for center proximity

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Get timestamp for the current frame
        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Perform YOLO detection
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        detections_with_boxes = []
        current_bboxes = [] # Store bboxes for drawing later if needed
        if results and results[0].boxes:
            for box in results[0].boxes:
                if box.xyxy is not None and len(box.xyxy) > 0 and box.cls is not None:
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    center = calculate_center(bbox)
                    detections_with_boxes.append((center, tuple(bbox))) # For tracker
                    current_bboxes.append(tuple(bbox)) # For potential drawing

                    # --- Check proximity to center line and record for receipt ---
                    if abs(center[0] - center_x_frame) < center_proximity_threshold:
                        try:
                            label_index = int(box.cls[0].cpu().item())
                            label_name = model.names[label_index]
                            center_items_receipt.append((timestamp_ms, label_name))
                            # print(f"Frame {frame_number}, Time {timestamp_ms:.2f}ms: Found '{label_name}' near center (x={center[0]})") # Debug print
                        except IndexError:
                            print(f"Warning: Could not get class label for a detection in frame {frame_number}")
                        except Exception as e:
                            print(f"Warning: Error processing label in frame {frame_number}: {e}")


        # Update tracker (still useful for visualization if needed)
        tracker.update(detections_with_boxes, frame_number)

        # --- Visualization ---
        if VISUALIZE:
            vis_frame = frame.copy() # Draw on a copy

            # Draw active tracks
            for track_id, track_data in tracker.tracks.items():
                if track_data['active'] and track_data['last_bbox'] is not None:
                    x1, y1, x2, y2 = track_data['last_bbox']
                    color = tracker._get_color(track_id)
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    # Draw track ID
                    label = f"ID: {track_id}"
                    cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Draw center point (optional)
                    # center_x, center_y = track_data['last_pos']
                    # cv2.circle(vis_frame, (center_x, center_y), 4, color, -1)

            # Display the frame
            cv2.imshow("YOLO Tracking", vis_frame)
            # Press 'q' to quit the video playback early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 print("Quitting visualization...")
                 break # Exit the loop for this video

        frame_number += 1

    cap.release()
    if VISUALIZE:
        cv2.destroyAllWindows() # Close the window for the current video
    print(f"Finished processing {os.path.basename(video_path)}. Found {len(center_items_receipt)} items near center.")

    # Return the list of items found near the center for this video
    return center_items_receipt


# --- Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to detect items near the center line and generate a receipt CSV.")
    parser.add_argument('--video_dir', type=str, default=VIDEO_DIR, help=f'Directory containing video files (default: {VIDEO_DIR})')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help=f'Directory to save the output CSV (default: {OUTPUT_DIR})')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help=f'Path to the YOLO model (default: {MODEL_PATH})')
    parser.add_argument('--conf_thresh', type=float, default=CONFIDENCE_THRESHOLD, help=f'Confidence threshold for detection (default: {CONFIDENCE_THRESHOLD})')
    parser.add_argument('--prox_thresh', type=float, default=MAX_DISTANCE_THRESHOLD, help=f'Pixel proximity threshold to center line (default: {MAX_DISTANCE_THRESHOLD})')
    parser.add_argument('--no_visualize', action='store_true', help='Disable video visualization')

    args = parser.parse_args()

    # Update configuration from args
    VIDEO_DIR = args.video_dir
    OUTPUT_DIR = args.output_dir
    MODEL_PATH = args.model_path
    CONFIDENCE_THRESHOLD = args.conf_thresh
    MAX_DISTANCE_THRESHOLD = args.prox_thresh # Allow overriding proximity threshold via args
    VISUALIZE = not args.no_visualize
    # Create output directory if it doesn't exist
    output_dir_path = os.path.abspath(OUTPUT_DIR) # Use absolute path for clarity
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"Created output directory: {output_dir_path}")

    # Load the YOLO model
    model_path_abs = os.path.abspath(MODEL_PATH)
    print(f"Loading model from {model_path_abs}...")
    try:
        yolo_model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model from {model_path_abs}: {e}")
        exit(1)

    # Find video files
    video_dir_abs = os.path.abspath(VIDEO_DIR)
    try:
        video_files = [f for f in os.listdir(video_dir_abs) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            print(f"Error: No video files found in {video_dir_abs}")
            exit(1)
    except FileNotFoundError:
        print(f"Error: Video directory not found: {video_dir_abs}")
        exit(1)


    print(f"Found {len(video_files)} video(s) in {video_dir_abs}")

    all_receipt_items = [] # Collect items from all videos

    # Process each video
    for video_file in video_files:
        video_full_path = os.path.join(video_dir_abs, video_file)
        # The JSON output is removed, process_video now returns the list
        receipt_items_for_video = process_video(video_full_path, yolo_model)
        all_receipt_items.extend(receipt_items_for_video)

    print(f"\nFinished processing all videos. Total items recorded: {len(all_receipt_items)}")

    # --- Write the combined receipt CSV ---
    if all_receipt_items:
        # Sort by timestamp before writing (optional but good practice)
        all_receipt_items.sort(key=lambda x: x[0])

        csv_output_filename = 'run.csv'
        csv_output_path = os.path.join(output_dir_path, csv_output_filename)

        try:
            with open(csv_output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'item']) # Write header
                # Convert timestamp from ms to seconds or keep as ms? Let's keep ms for precision.
                writer.writerows(all_receipt_items)
            print(f"Receipt saved successfully to: {csv_output_path}")
        except Exception as e:
            print(f"Error writing receipt CSV to {csv_output_path}: {e}")
    else:
        print("No items were found near the center line in any video.")


    if VISUALIZE:
        cv2.destroyAllWindows() # Ensure all windows are closed at the end
