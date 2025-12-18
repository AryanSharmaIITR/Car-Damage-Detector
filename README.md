# 🚗 Car Damage Detection System

A deep learning-based web application that automatically detects and classifies car damage from uploaded images. The system uses a ResNet-50 model fine-tuned on car damage datasets to identify different types of damage.

## ✨ Features

- **Six Damage Classes**: Classifies car images into:
  - Front Breakage
  - Front Crushed
  - Front Normal (no damage)
  - Rear Breakage
  - Rear Crushed
  - Rear Normal (no damage)
  
- **User-Friendly Interface**: Clean Streamlit-based web interface
- **Real-time Predictions**: Instant damage classification with confidence scores
- **Visual Feedback**: Displays uploaded image alongside prediction results
- **Detailed Probabilities**: Shows confidence scores for all damage classes

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/car-damage-detector.git
cd car-damage-detector