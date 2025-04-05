# YOLO Video Inference & Annotation

This folder contains Python scripts for:
- Running object detection and tracking on images or videos using YOLO  
- Semi-automated annotation for refining training data  
- Utility functions for video and file I/O  

## File Descriptions

### 1. `yolo_inference.py`
- **Purpose**: Run inference on single images, whole directories, or video files (including webcams).
- **Usage**:
  ```bash
  python yolo_inference.py \
      --source ../data/video/myvideo.mp4 \
      --weights ../models/yolo/best.pt \
      --conf 0.5 \
      --show \
      --save