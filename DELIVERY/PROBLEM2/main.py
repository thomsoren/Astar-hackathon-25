import cv2
import os
import json
import math
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# --- Configuration ---
MODEL_PATH = '../../trained_models/weights/best.pt' # Relative path from main.py
VIDEO_DIR = '../../data/videos'           # Relative path from main.py
OUTPUT_DIR = 'output'                     # Relative path from main.py
MAX_DISTANCE_THRESHOLD = 50               # Max pixel distance to match a detection to a track
MAX_FRAMES_UNSEEN = 10                    # Max frames to keep a track alive without detection
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
        # Store track_id -> {coords: list, last_pos: tuple, last_bbox: tuple, unseen_frames: int, active: bool, class_id: int, class_name: str}
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

    def update(self, detections_with_boxes_and_classes, frame_number):
        """
        Updates tracks based on new detections.
        detections_with_boxes_and_classes: list of tuples [(center_x, center_y), (x1, y1, x2, y2), class_id, class_name]
        """
        active_tracks = self._get_active_tracks()
        matched_track_ids = set()
        used_detection_indices = set()

        # Try to match detections to existing active tracks
        if detections_with_boxes_and_classes and active_tracks:
            # Build cost matrix (distances between centers)
            cost_matrix = np.full((len(active_tracks), len(detections_with_boxes_and_classes)), float('inf'))
            track_ids = list(active_tracks.keys())
            detection_centers = [det[0] for det in detections_with_boxes_and_classes] # Extract centers

            for i, track_id in enumerate(track_ids):
                track_pos = active_tracks[track_id]['last_pos']
                for j, (det_center, _, _, _) in enumerate(detections_with_boxes_and_classes):
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
                    det_center, det_bbox, class_id, class_name = detections_with_boxes_and_classes[det_idx]
                    self.tracks[track_id]['coords'].append([frame_number, det_center[0], det_center[1]])
                    self.tracks[track_id]['last_pos'] = det_center
                    self.tracks[track_id]['last_bbox'] = det_bbox # Store the bbox
                    self.tracks[track_id]['unseen_frames'] = 0
                    self.tracks[track_id]['class_id'] = class_id # Update class info
                    self.tracks[track_id]['class_name'] = class_name
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
        for i, (det_center, det_bbox, class_id, class_name) in enumerate(detections_with_boxes_and_classes):
            if i not in used_detection_indices:
                new_id = self.next_track_id
                self.tracks[new_id] = {
                    'coords': [[frame_number, det_center[0], det_center[1]]],
                    'last_pos': det_center,
                    'last_bbox': det_bbox, # Store initial bbox
                    'unseen_frames': 0,
                    'active': True,
                    'class_id': class_id,
                    'class_name': class_name
                }
                self.next_track_id += 1

    def get_results(self):
        """Returns the coordinate history for all tracks (active and inactive)."""
        # Filter out tracks that might have been created but never updated significantly (optional)
        # return {tid: track['coords'] for tid, track in self.tracks.items() if len(track['coords']) > 1}
        return {tid: track['coords'] for tid, track in self.tracks.items()}


# --- Main Processing Function ---
def process_video(video_path, model, output_file):
    """Processes a single video file for object tracking."""
    print(f"Processing video: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    tracker = Tracker(max_distance=MAX_DISTANCE_THRESHOLD, max_unseen=MAX_FRAMES_UNSEEN)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Perform YOLO detection
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        detections_with_boxes_and_classes = []
        current_bboxes = [] # Store bboxes for drawing later if needed
        if results and results[0].boxes:
            for box in results[0].boxes:
                if box.xyxy is not None and len(box.xyxy) > 0:
                    bbox = box.xyxy[0].cpu().numpy().astype(int) # Use int for drawing
                    center = calculate_center(bbox)
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    detections_with_boxes_and_classes.append((center, tuple(bbox), class_id, class_name)) # Store center, bbox, and class info
                    current_bboxes.append(tuple(bbox))

        # Update tracker
        tracker.update(detections_with_boxes_and_classes, frame_number)

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
                    # Draw track ID and class name
                    label = f"ID: {track_id} - {track_data['class_name']}"
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
    print(f"Finished processing {os.path.basename(video_path)}. Found {len(tracker.tracks)} tracks.")

    # Save results
    tracking_data = tracker.get_results()
    try:
        with open(output_file, 'w') as f:
            json.dump(tracking_data, f, indent=4)
        print(f"Tracking results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")


# --- Script Execution ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Load the YOLO model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        yolo_model = YOLO(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        exit()

    # Find video files
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        print(f"Error: No video files found in {VIDEO_DIR}")
        exit()

    print(f"Found {len(video_files)} video(s) in {VIDEO_DIR}")

    # Process each video
    for video_file in video_files:
        video_full_path = os.path.join(VIDEO_DIR, video_file)
        output_filename = f"{os.path.splitext(video_file)[0]}_tracks.json"
        output_full_path = os.path.join(OUTPUT_DIR, output_filename)
        process_video(video_full_path, yolo_model, output_full_path)

    print("All videos processed.")
    if VISUALIZE:
        cv2.destroyAllWindows() # Ensure all windows are closed at the end