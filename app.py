# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import gradio as gr

# ---- Model definition (must match training) ----
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.fc1   = nn.Linear(32 * 56 * 56, 112)
        self.drop1 = nn.Dropout(0.5)
        self.fc2   = nn.Linear(112, 84)
        self.drop2 = nn.Dropout(0.2)
        self.fc3   = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# ---- Load model once (CPU is fine for Spaces unless you enable GPU) ----
MODEL_PATH = "pneumonia_model.pth"
device = torch.device("cpu")

model = Net()
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# ---- Preprocess must match training ----
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

CLASSES = ["Normal", "Pneumonia"]

def predict(img_pil: Image.Image):
    img = img_pil.convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()
    return {CLASSES[i]: float(probs[i]) for i in range(2)}

# ---- Gradio interface ----
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Chest X-ray"),
    outputs=gr.Label(num_top_classes=2),
    title="Pneumonia Detector (CNN)",
    description="Upload a chest X-ray image. The model predicts Normal vs Pneumonia with probabilities."
)

demo.launch()
