from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Create FastAPI app
app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = "model/plant_disease_model.h5"
LABELS_PATH = "model/labels.json"
UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    labels = json.load(f)

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Preprocess image
    img_array = preprocess_image(file_path)

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return {
        "prediction": labels[str(class_idx)],
        "confidence": round(confidence * 100, 2)
    }

@app.get("/")
def root():
    return {"message": "Plant Disease Classification API is running"}
