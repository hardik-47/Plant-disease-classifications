from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import uvicorn
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------
# 1. Initialize FastAPI app
# -------------------------
app = FastAPI()

# Enable CORS (so frontend JS can call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 2. Load Model + Class Names
# -------------------------
MODEL_PATH = "model/plant_disease_model2.keras"   # use .h5 (what you saved)
CLASS_NAMES_PATH = "class_names2.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

IMG_SIZE = 224  # MobileNetV2 input size

# -------------------------
# 3. Preprocessing Function
# -------------------------
def preprocess_image(image: Image.Image):
    # Convert to RGB (ensure 3 channels)
    image = image.convert("RGB")

    # Convert to tensor
    img = tf.convert_to_tensor(np.array(image), dtype=tf.float32)

    # Resize to (224, 224)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)

    # Apply same preprocessing as training
    img = preprocess_input(img)
    return img


# -------------------------
# 4. Prediction Endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load image
        image = Image.open(file.file)
        
        # Preprocess
        img = preprocess_image(image)
        
        # Predict
        preds = model.predict(img)
        class_id = int(np.argmax(preds[0]))
        confidence = float(preds[0][class_id])

        # preds = model.predict(img)
        print("Raw Predictions:", preds)        # see all probabilities
        print("Sum of probs:", np.sum(preds))   # should ≈ 1

        
        return JSONResponse({
            "prediction": class_names[class_id],
            "confidence": round(confidence, 2)
        })
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -------------------------
# 5. Run the app (for local dev)
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





# from google.colab import drive
# drive.mount('/content/drive')
# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     "/content/drive/MyDrive/Plant",
#     seed=123,
#     shuffle=True,
#     image_size=(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size=BATCH_SIZE
# )