import gradio as gr
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os # To construct absolute path for model

# --- Configuration ---
# Construct the absolute path to the model file relative to this script's location or CWD
# Assuming the script runs from the project root '/Users/nybruker/Documents/Hackaton/A-star25'
MODEL_PATH = os.path.join(os.getcwd(), 'DELIVERY/PROBLEM1/src/yolo11m.pt')

# Load the YOLO model (ensure the path is correct)
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLO model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    # Fallback or raise error if model is critical
    model = None

# Class names from dataset.yaml
CLASS_NAMES = {
  0: "Bananer Bama", 1: "Epler Røde", 2: "Paprika Rød", 3: "Appelsin",
  4: "Bananer Økologisk", 5: "Red Bull Regular 250ml boks", 6: "Red Bull Sukkerfri 250ml boks",
  7: "Karbonadedeig 5% u/Salt og Vann 400g Meny", 8: "Kjøttdeig Angus 14% 400g Meny",
  9: "Ruccula 65g Grønn&Frisk", 10: "Rundstykker Grove Fullkorn m/Frø Rustikk 6stk 420g",
  11: "Leverpostei Ovnsbakt Orginal 190g Gilde", 12: "Kokt Skinke Ekte 110g Gilde",
  13: "Yoghurt Skogsbær 4x150g Tine", 14: "Norvegia 26% skivet 150g Tine",
  15: "Jarlsberg 27% skivet 120g Tine", 16: "Cottage Cheese Mager 2% 400g Tine",
  17: "Yt Protein Yoghurt Vanilje 430g Tine", 18: "Frokostegg Frittgående L 12stk Prior",
  19: "Gulrot 750g Beger", 20: "Gulrot 1kg pose First Price", 21: "Evergood Classic Filtermalt 250g",
  22: "Pepsi Max 0,5l flaske", 23: "Frokostyoghurt Skogsbær 125g pose Q",
  24: "Original Havsalt 190g Sørlandschips", 25: "Kvikk Lunsj 3x47g Freia"
}

# --- Food Detection Function ---
def your_food_detection_function(image_input):
    """
    Processes the uploaded image with the YOLO model.
    Returns the annotated image for gr.HighlightedImage and a text summary.
    """
    if image_input is None:
        print("No image provided.")
        return None, "Please upload an image first."
    if model is None:
        print("YOLO model not loaded.")
        return image_input, "Error: Model not available." # Return original image if model failed

    print("Processing image for food detection...")
    try:
        # Perform prediction
        results = model(image_input, verbose=False) # verbose=False to reduce console clutter

        # Check if results are valid and contain boxes
        if not results or not hasattr(results[0], 'boxes'):
             print("No detections found or invalid results format.")
             return image_input, "No food items detected." # Return original image

        # Extract bounding boxes, confidences, and class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (x_min, y_min, x_max, y_max)
        confidences = results[0].boxes.conf.cpu().numpy() # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int) # Class IDs

        highlighted_boxes = []
        detected_items_list = []

        if len(boxes) > 0:
            print(f"Detected {len(boxes)} items.")
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                if cls_id in CLASS_NAMES:
                    label = CLASS_NAMES[cls_id]
                    label_with_conf = f"{label} ({conf:.2f})"

                    # Format for gr.HighlightedImage: (bounding_box, label)
                    # Bounding box: [x_min, y_min, x_max, y_max] in pixel values
                    highlighted_boxes.append((box.tolist(), label_with_conf))

                    # Format for text summary
                    detected_items_list.append(f"- {label}: {conf:.2f}")
                else:
                    print(f"Warning: Detected class ID {cls_id} not in CLASS_NAMES.")

            annotated_image_output = (image_input, highlighted_boxes)
            detected_items_text = "Detected Items:\n" + "\n".join(detected_items_list) if detected_items_list else "No food items detected."
        else:
            print("No food items detected in the image.")
            annotated_image_output = (image_input, []) # Return original image with empty boxes
            detected_items_text = "No food items detected."

        print("Food detection complete.")
        return annotated_image_output, detected_items_text

    except Exception as e:
        print(f"Error during YOLO prediction: {e}")
        # Return original image and error message
        return image_input, f"Error during processing: {e}"

# --- Theft/Scanning Detection Function ---
def your_video_detection_function(video_path):
    """
    Processes the video for theft/scanning detection (placeholder for action detection).
    Uses YOLO to detect items in a sample frame and compares with a dummy receipt.
    Returns the original video path, a comparison DataFrame, and a status log.
    """
    print(f"Processing video: {video_path}")
    if video_path is None:
        print("No video provided.")
        return None, pd.DataFrame(columns=["Item", "Detected in Video", "On Receipt", "Discrepancy"]), "Please upload a video first."
    if model is None:
        print("YOLO model not loaded.")
        # Return original video path, empty dataframe, and error message
        return video_path, pd.DataFrame(columns=["Item", "Detected in Video", "On Receipt", "Discrepancy"]), "Error: Item detection model not available."

    # --- Placeholder: Action Detection ---
    # In a real scenario, analyze the video for "steal" vs "no steal" actions.
    # For this demo, we'll just assume an action was detected.
    action_detected = "Scan" # or "Potential Theft" based on a real model

    # --- Item Detection using YOLO (on a sample frame) ---
    detected_video_items = []
    try:
        print("Extracting a sample frame for item detection...")
        # Use a library like OpenCV to extract a frame if needed.
        # For simplicity, let's assume the model can process the video path directly
        # or we extract a frame manually (not implemented here).
        # We'll run YOLO on a dummy frame or the first frame if possible.
        # NOTE: Running YOLO on the entire video is computationally expensive for a demo.
        # Let's simulate detection results based on the model.
        
        # Simulate running model on one frame (replace with actual frame extraction and prediction)
        # For demo purposes, let's reuse some dummy logic similar to image detection
        # In reality, you'd extract a frame: cap = cv2.VideoCapture(video_path); ret, frame = cap.read(); cap.release()
        # Then run: results = model(frame)
        
        # --- Dummy YOLO results for video ---
        # Replace this section with actual frame extraction and YOLO prediction
        print("Simulating YOLO detection on a video frame...")
        simulated_results = { # Dummy data structure
            "boxes": [[10, 10, 50, 50], [60, 60, 100, 100]],
            "conf": [0.85, 0.78],
            "cls": [1, 22] # Corresponds to 'Epler Røde' and 'Pepsi Max'
        }
        
        if simulated_results and len(simulated_results.get("cls", [])) > 0:
             for cls_id, conf in zip(simulated_results["cls"], simulated_results["conf"]):
                 if cls_id in CLASS_NAMES:
                     detected_video_items.append(CLASS_NAMES[cls_id])
                     print(f"  Detected: {CLASS_NAMES[cls_id]} (Conf: {conf:.2f})")
                 else:
                     print(f"  Warning: Detected class ID {cls_id} not in CLASS_NAMES.")
        else:
             print("No items detected in the sample frame.")
             
        # --- Placeholder: Video Annotation ---
        # A real implementation would draw bounding boxes and action labels onto the video frames.
        # For this demo, we return the original video path.
        annotated_video_path = video_path
        print(f"Detected items in video (sample frame): {detected_video_items}")

    except Exception as e:
        print(f"Error during video item detection: {e}")
        status_log = f"Error during item detection: {e}"
        # Return original video, empty dataframe, and status
        return video_path, pd.DataFrame(columns=["Item", "Detected in Video", "On Receipt", "Discrepancy"]), status_log

    # --- Receipt Comparison ---
    # Dummy receipt data (replace with actual data source if available)
    receipt_items = ["Epler Røde", "Bananer Bama", "Pepsi Max 0,5l flaske"]
    print(f"Dummy Receipt Items: {receipt_items}")

    # Combine detected and receipt items for comparison
    all_items = sorted(list(set(detected_video_items + receipt_items)))
    comparison_data = {
        "Item": [],
        "Detected in Video": [],
        "On Receipt": [],
        "Discrepancy": []
    }

    discrepancy_count = 0
    for item in all_items:
        detected = item in detected_video_items
        on_receipt = item in receipt_items
        discrepancy = detected != on_receipt
        comparison_data["Item"].append(item)
        comparison_data["Detected in Video"].append("✔️" if detected else "❌")
        comparison_data["On Receipt"].append("✔️" if on_receipt else "❌")
        comparison_data["Discrepancy"].append("⚠️ Yes" if discrepancy else "✅ No")
        if discrepancy:
            discrepancy_count += 1

    receipt_comparison_df = pd.DataFrame(comparison_data)

    # --- Status Log ---
    status_log = (
        f"Video processing complete.\n"
        f"Action Detected (Placeholder): {action_detected}\n"
        f"Items Detected (Sample Frame): {len(detected_video_items)}\n"
        f"Items on Receipt (Dummy): {len(receipt_items)}\n"
        f"Discrepancies Found: {discrepancy_count}"
    )
    print("Video processing and comparison complete.")

    return annotated_video_path, receipt_comparison_df, status_log

# Define the Gradio interface using Blocks and Tabs
with gr.Blocks(title="DataVision Food & Theft Detection") as demo:
    gr.Markdown("# DataVision Food & Theft Detection") # Header/Title

    with gr.Tabs():
        # Tab 1: Food Detection
        with gr.Tab("Food Detection"):
            gr.Markdown("## Task 1: Food Detection in Images")
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Upload Food Image", type="numpy") # Use numpy for potential processing
                    detect_button = gr.Button("Detect Food")
                with gr.Column(scale=1):
                    # Revert to gr.AnnotatedImage as gr.HighlightedImage doesn't exist
                    annotated_output = gr.AnnotatedImage(label="Detection Result")
                    status_output = gr.Label(label="Detected Items") # Using Label for simple text output

            # Optional Collapsible Training Status (Placeholder)
            with gr.Accordion("Show Training Status (Optional)", open=False):
                 training_status = gr.Textbox("Training status updates would appear here...", label="Training Log", interactive=False)

            # Wire the button click event (update output component)
            detect_button.click(fn=your_food_detection_function,
                                inputs=image_input,
                                outputs=[annotated_output, status_output]) # Reverted output component name

        # Tab 2: Theft/Scanning Detection
        with gr.Tab("Theft/Scanning Detection"):
            gr.Markdown("## Task 2: Theft/Scanning Detection at Self-Checkout")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Self-Checkout Video")
                    process_button = gr.Button("Process Video")
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Detection Output", interactive=False) # Updated label
                    log_output = gr.Textbox(label="Status Log", interactive=False)

            with gr.Row():
                 receipt_comparison = gr.Dataframe(label="Receipt Comparison", interactive=False, wrap=True)


            # Wire the button click event
            process_button.click(fn=your_video_detection_function,
                                 inputs=video_input,
                                 outputs=[video_output, receipt_comparison, log_output])

    # Footer (Optional)
    gr.Markdown("---")
    gr.Markdown("Developed for the A-star25 Hackathon.")


# Launch the app
if __name__ == "__main__":
    # Running on 0.0.0.0 makes it accessible on the network
    # Running on a specific port like 7861
    # Enable queue for handling multiple requests if needed
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False) # share=True generates a public link if needed
