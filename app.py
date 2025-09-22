from fastapi import FastAPI, UploadFile, File, Response, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
import torch.nn as nn
import os

app = FastAPI()

LABELS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
          "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
          "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
          "Pleural Other", "Fracture", "Support Devices"]

device = torch.device("cpu")

# -----------------------------
# ✅ Load DenseNet model
# -----------------------------
def load_model(path, device):
    # Use new torchvision API (no warnings)
    model = models.densenet121(weights=None)  # same as pretrained=False
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(LABELS))

    # Load fine-tuned checkpoint
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("classifier.0."):
            new_key = k.replace("classifier.0.", "classifier.")
            new_state_dict[new_key] = v
        elif not k.startswith("classifier."):
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

model = load_model("finetuned_model.pth", device)

# -----------------------------
# ✅ Transform for input images
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# ✅ Root endpoints
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome! FastAPI app is running on Render 🚀"}

@app.head("/")
def head_root():
    return Response(status_code=200)

@app.post("/")
def post_root(request: Request):
    return {"error": "POST not supported on root"}

# -----------------------------
# ✅ Health check (with model test)
# -----------------------------
@app.get("/health", include_in_schema=False)
def health_check():
    try:
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy)
        return {"status": "ok", "model": "loaded"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "details": str(e)})

# -----------------------------
# ✅ Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        results = {label: round(float(prob) * 100, 2) for label, prob in zip(LABELS, probs)}

        return {"probabilities": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------------
# ✅ Static & favicon
# -----------------------------
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)  # Empty response if no favicon present
