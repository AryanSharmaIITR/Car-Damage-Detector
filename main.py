import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np

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
    """Convert image for display without prediction preprocessing"""
    image = Image.open(image_file).convert("RGB")
    # Simply resize the image for display
    image = image.resize((224, 224))
    return image


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
    # Display the uploaded image
    display_image = preprocess_for_display(uploaded_image)

    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(display_image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    label, confidence = predict(uploaded_image)

    with col2:
        st.markdown("### Prediction Results")
        st.success(f"**Class:** {label}")
        st.info(f"**Confidence:** {confidence:.2%}")

        # Optional: Show confidence for all classes
        st.markdown("### All Class Probabilities:")

        # We need to recalculate all probabilities
        image = Image.open(uploaded_image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        image_tensor = transform(image).unsqueeze(0)
        model = load_model()

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)

        # Display probabilities for all classes
        for i, class_name in enumerate(CLASS_NAMES):
            prob = probs[0, i].item()
            st.progress(prob, text=f"{class_name}: {prob:.2%}")

else:
    st.info("Please upload a car image to begin.")