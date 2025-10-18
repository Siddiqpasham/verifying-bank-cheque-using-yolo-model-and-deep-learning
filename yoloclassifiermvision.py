import cv2
import torch
import os
import re
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load YOLO model
model = YOLO("best.pt")  # Update with your trained model path

# Load TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

# Load the cheque image
image_path = "train/images/C2.jpg"  # Update with your cheque image path
image = cv2.imread(image_path)

# Run YOLO detection
results = model(image)

# Create a directory to store cropped images
output_dir = "cropped_regions"
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store extracted text
extracted_data = {}

# Function to preprocess images for better OCR accuracy
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce noise using Bilateral Filter
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Morphological operations (dilate to enhance text)
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)

    return processed

# Function to extract text using TrOCR
def extract_text_trocr(image):
    # Convert OpenCV image to PIL format
    pil_image = Image.fromarray(image).convert("RGB")

    # Process image using TrOCR processor
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values

    # Perform OCR using TrOCR model
    generated_ids = ocr_model.generate(pixel_values)
    extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return extracted_text.strip()

# Function to clean text based on field type
def clean_text(label, text):
    text = text.strip()

    if label == "accno":  
        # Extract only digits (remove unnecessary words)
        text = re.sub(r"\D", "", text)
    
    elif label == "numbers":  
        # Extract only numeric values (remove currency symbols, words)
        text = re.search(r"\d+", text)
        text = text.group() if text else ""

    elif label == "words":
        # Remove unwanted characters
        text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text

# Process detections
for result in results:
    for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        label = model.names[int(cls)]  # Get class label
        
        # Crop the detected region
        cropped_img = image[y1:y2, x1:x2]
        
        # Save cropped image
        cropped_path = os.path.join(output_dir, f"{label}.jpg")
        cv2.imwrite(cropped_path, cropped_img)

        # Preprocess the cropped image for better OCR
        processed_image = preprocess_image(cropped_img)

        # Extract text using TrOCR
        text = extract_text_trocr(processed_image)

        # Clean extracted text
        cleaned_text = clean_text(label, text)

        # Store the extracted text
        extracted_data[label] = cleaned_text

# Display extracted and cleaned information
for key, value in extracted_data.items():
    print(f"{key}: {value}")
