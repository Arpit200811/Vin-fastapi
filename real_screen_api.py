from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
BASE_PATH = r"C:\Users\DELL\OneDrive\Desktop\dataset"
MODEL_PATH = os.path.join(BASE_PATH, "vin_model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)
app = FastAPI()
def preprocess_image(img: Image.Image):
    """
    Preprocess uploaded image so it matches model input (128x128x3).
    Converts RGB → BGR since model was trained with OpenCV (BGR).
    """
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = img_array[:, :, ::-1]
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
