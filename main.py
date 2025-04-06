import os
import joblib
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model & label map
model = joblib.load("model/vitamin_model.pkl")
label_map = joblib.load("model/label_map.pkl")
print("✅ Loaded label_map:", label_map)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, image: UploadFile = File(...)):
    image_path = f"static/uploads/{image.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        img = Image.open(image_path).convert("RGB").resize((64, 64))
        img_array = np.array(img) / 255.0
        features = img_array.mean(axis=(0, 1)).reshape(1, -1)

        # Debugging output
        pred = model.predict(features)[0]
        print("🧠 Predicted class:", pred)
        print("🔍 Available labels in label_map:", label_map)

        pred_label = label_map.get(pred, "Unknown deficiency")

        result = f"According to this image, you may be facing <b>{pred_label}</b>."
    except Exception as e:
        result = f"Prediction failed: {str(e)}"

    return templates.TemplateResponse("result.html", {"request": request, "result": result})
