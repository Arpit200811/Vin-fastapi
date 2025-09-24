import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
IMG_SIZE = 128
BASE_PATH = r"C:\Users\DELL\OneDrive\Desktop\dataset"
MODEL_PATH = os.path.join(BASE_PATH, "vin_model.h5")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    sys.exit()

# Load model
model = load_model(MODEL_PATH)

# Check for image path argument
if len(sys.argv) < 2:
    print("⚠️ Usage: python test_vin_model.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

# Check if image exists
if not os.path.exists(image_path):
    print(f"❌ Error: File '{image_path}' not found!")
    sys.exit()

# Read and preprocess image
img = cv2.imread(image_path)
if img is None:
    print(f"❌ Error: Unable to read image '{image_path}'")
    sys.exit()

img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_normalized = img_resized / 255.0
img_input = np.expand_dims(img_normalized, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_input)
prob = float(prediction[0][0])

# Output result
if prob >= 0.5:
    print(f"{image_path} => ✅ Real VIN (Confidence: {prob*100:.2f}%)")
else:
    print(f"{image_path} => ❌ Fake VIN (Confidence: {(1-prob)*100:.2f}%)")
