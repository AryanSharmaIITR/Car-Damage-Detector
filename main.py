import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
from matplotlib import pyplot as plt
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']
trained_model = None


# Load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True            
            
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x
def show(image_path):
    image = Image.open(image_path).convert("RGB")
    # Image transformation for input preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor_show=transform(image).squeeze(0)
    return image_tensor_show

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    # Image transformation for input preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)
    global trained_model

    if not trained_model:
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load("saved_model.pth"))
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()],


st.title("Car Damage Detection")

input_image=st.file_uploader("Upload Your Car Pic", type=["jpg", "jpeg", "png"])

if input_image:
    image_tensor=show(input_image)
    plt.figure(figsize=(4,4))
    plt.imshow(image_tensor.permute(1,2,0))
    plt.axis("off")
    st.pyplot(plt)
    st.success(f"This Car is Belong to: {predict(input_image)[0]}")
else:
    st.write("Pls Upload")