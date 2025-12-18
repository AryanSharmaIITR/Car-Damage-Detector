import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt

CLASS_NAMES = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super().__init__()

        self.model = models.resnet50(
            weights=ResNet50_Weights.DEFAULT
        )
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    model = CarClassifierResNet()
    model.load_state_dict(
        torch.load("saved_model.pth", map_location="cpu")
    )
    model.eval()
    return model

# -----------------------------------
# Image Display (NO normalization)
# -----------------------------------
def preprocess_for_display(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image)

def predict(image_file):
    image = Image.open(image_file).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    model = load_model()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return CLASS_NAMES[pred_idx], confidence

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="Car Damage Detection", layout="centered")
st.title("🚗 Car Damage Detection")

uploaded_image = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    img_tensor = preprocess_for_display(uploaded_image)
    plt.figure(figsize=(12, 12))
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.axis("off")
    st.pyplot(plt)

    label, confidence = predict(uploaded_image)

    st.success(
        f"**Prediction:** {label}\n\n"
        f"**Confidence:** {confidence:.2%}"
    )

else:
    st.info("Please upload a car image to begin.")
