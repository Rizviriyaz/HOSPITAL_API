import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# List of labels from your dataset
LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

# Load model
def load_model(model_path, device):
    model = models.densenet121(pretrained=False)
    num_ftrs = model.classifier.in_features

    # Make sure this matches training
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, len(LABELS))
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

# Preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(model, image_path, device, threshold=0.3):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    results = {label: float(prob) for label, prob in zip(LABELS, probs)}
    positive = {k: v for k, v in results.items() if v > threshold}
    return {"probabilities": results, "positive_findings": positive}
