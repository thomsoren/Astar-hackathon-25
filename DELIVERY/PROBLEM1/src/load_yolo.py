from ultralytics import YOLO

def run_yolo():
    # Load pre-trained YOLOv11 nano model
    model = YOLO('yolo11m.pt')  # You can replace with yolo11s.pt, yolo11m.pt, etc.

    # Run inference on an image
    results = model('path/to/your/image.jpg')  # replace with actual image path

    # Show results (bounding boxes, labels)
    results.show()

if __name__ == "__main__":
    run_yolo()