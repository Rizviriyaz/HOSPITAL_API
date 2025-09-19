# app.py
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# ----------------------------
# Setup FastAPI
# ----------------------------
app = FastAPI(title="Chest X-ray Diagnosis API")

# ----------------------------
# Model Definition
# ----------------------------
num_classes = 14  # same as finetune.py (DEFAULT_LABELS length)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=None)
# ✅ Match the training definition (Sequential → Linear)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, num_classes)
)

# Load checkpoint (trained model)
checkpoint = torch.load("finetuned_model.pth", map_location=device)
# If you saved only model.state_dict()
if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# ----------------------------
# Preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return {"message": "Chest X-ray Diagnosis API is running ✅"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        # Convert to dict {label: probability}
        results = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

        return JSONResponse(content={"predictions": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
