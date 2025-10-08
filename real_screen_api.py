from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ya specific domain(s)
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Model Path Handling
# -----------------------------
# Local path (for your system)
LOCAL_MODEL_PATH = r"C:\Users\DELL\OneDrive\Desktop\dataset\vin_model.h5"

# Project-relative path (for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_MODEL_PATH = os.path.join(BASE_DIR, "models", "vin_model.h5")

# Optionally, a remote link if model is large
MODEL_URL = "https://your-storage-link/vin_model.h5"  # optional (if >100MB)

# Determine which path to use
if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
elif os.path.exists(RENDER_MODEL_PATH):
    MODEL_PATH = RENDER_MODEL_PATH
else:
    # Auto-download model if not found
    print("Model not found locally. Attempting to download...")
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    try:
        r = requests.get(MODEL_URL)
        if r.status_code == 200:
            with open(RENDER_MODEL_PATH, "wb") as f:
                f.write(r.content)
            MODEL_PATH = RENDER_MODEL_PATH
            print("✅ Model downloaded successfully.")
        else:
            raise FileNotFoundError(f"Failed to download model. Status: {r.status_code}")
    except Exception as e:
        raise FileNotFoundError(f"Model not found locally or remotely: {e}")

print(f"🔹 Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")


# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_image(img: Image.Image):
    """
    Preprocess uploaded image so it matches model input (128x128x3).
    Converts RGB → BGR since model was trained with OpenCV (BGR).
    """
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array[:, :, ::-1]  # RGB → BGR
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


# -----------------------------
# API Routes
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        input_array = preprocess_image(img)

        pred = model.predict(input_array)[0][0]  # sigmoid output
        label = "real" if pred >= 0.5 else "fake"
        prob = float(pred if label == "real" else 1 - pred)

        return JSONResponse({"label": label, "prob": prob})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def read_root():
    return {"message": "VIN Model API is running 🚀"}
