# FastAPI app that loads the trained pneumonia detection model
# and exposes a /predict endpoint for image classification.

import sys
import os
import io

import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# Allow importing model.py from the src/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import get_resnet_model

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_resnet_adam.pt")
CLASSES = ["Covid", "Normal", "Viral Pneumonia"]
CONFIDENCE_THRESHOLD = 0.60   # below this, we say "uncertain" instead of forcing an answer
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cpu")  # Cloud Run free tier is CPU-only, which is fine for this model size

app = FastAPI(title="Pneumonia Detection API")

# Allow requests from any origin (needed so our Vercel frontend, on a
# different domain, is allowed to call this API from the browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Load model once at startup (not per-request - that would be slow)
# ----------------------------
model = get_resnet_model(num_classes=len(CLASSES), freeze_backbone=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


@app.get("/")
def health_check():
    # Simple endpoint to confirm the API is alive
    return {"status": "ok", "message": "Pneumonia Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image bytes and convert to RGB
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess exactly the same way as during training/testing
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # add batch dimension

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]  # convert to 0-1 confidence scores

    top_prob, top_idx = torch.max(probabilities, dim=0)
    top_class = CLASSES[top_idx.item()]
    confidence = top_prob.item()

    all_probs = {CLASSES[i]: round(probabilities[i].item(), 4) for i in range(len(CLASSES))}

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "Uncertain",
            "confidence": round(confidence, 4),
            "message": "Model is not confident this is a clear chest X-ray match. Please upload a clear chest X-ray image.",
            "all_probabilities": all_probs,
        }

    return {
        "prediction": top_class,
        "confidence": round(confidence, 4),
        "message": None,
        "all_probabilities": all_probs,
    }

    