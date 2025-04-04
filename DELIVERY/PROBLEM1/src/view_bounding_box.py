import cv2
import pandas as pd
import os

def draw_bounding_boxes(image_path, predictions):
    """Draws bounding boxes on the image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    tokens = predictions.split()
    if len(tokens) % 6 != 0:
        print("Warning: The prediction string doesn't have a multiple of 6 tokens.")
    
    for i in range(0, len(tokens), 6):
        try:
            class_label = tokens[i]
            confidence = float(tokens[i+1])
            x1 = int(tokens[i+2])
            y1 = int(tokens[i+3])
            x2 = int(tokens[i+4])
            y2 = int(tokens[i+5])
        except IndexError:
            print("Incomplete prediction found; skipping.")
            continue

        # Validate and correct the coordinate order if necessary
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label
        label = f"{class_label} {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Create a resizable window to better view the image
    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_submission(csv_file, images_folder):
    """Reads the submission file and processes each image."""
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        image_id = row['image_id']
        predictions = row['PredictionString']
        
        image_path = os.path.join(images_folder, image_id)
        if os.path.exists(image_path):
            draw_bounding_boxes(image_path, predictions)
        else:
            print(f"Image not found: {image_path}")

if __name__ == "__main__":
    csv_file = "submission.csv"
    images_folder = "data/images/test"
    process_submission(csv_file, images_folder)
