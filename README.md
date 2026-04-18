---
title: Vehicle Damage Detection
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# 🚗 Vehicle Damage Detection & Scanner

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-blueviolet?style=for-the-badge)](https://ultralytics.com/)
[![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey?style=for-the-badge&logo=flask&logoColor=black)](https://flask.palletsprojects.com/)

A comprehensive deep learning system for automated vehicle damage assessment. This project includes high-performance object detection models (YOLOv8, EfficientDet) trained on the **CarDD** dataset and a user-friendly web application for real-time damage scanning.

---

## ✨ Features

- 🏎️ **Dual Architecture Support**: Training scripts for both YOLOv8 (Ultralytics) and Faster R-CNN (Torchvision).
- 🔍 **Deep Inspection**: Detects up to 6 types of damage: *Dent, Scratch, Crack, Shattered Glass, Bumper Damage, and Deformation*.
- 🌐 **Web UI Interface**: Interactive "Scanner App" for uploading car photos and visualizing results instantly.
- 📈 **Metric Tracking**: Automated CSV logging for mAP, precision, and recall across training experiments.
- ⚙️ **Windows Optimized**: Built-in support for Windows-specific training constraints (VRAM management, worker handling).

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd dl
```

### 2. Set Up Environment
It is recommended to use a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 📊 Model Training

#### YOLOv8m (High Performance)
To train the YOLOv8 model on the CarDD dataset:
```bash
python train_yolov8.py
```
*Weights and metrics will be saved to `runs/yolov8m/weights/`.*

#### Faster R-CNN (ResNet-50 FPN)
To train the Faster R-CNN architecture:
```bash
python train_rcnn_v2.py
```
*This script uses a ResNet-50 backbone with FPN, optimized for precise bounding box detection.*

---

### 💻 Scanner Web Application

The web application provides a visual interface for the trained models.

1. **Start the server**:
   ```bash
   python scanner_app/app.py
   ```
2. **Access the application**:
   Open your browser and navigate to `http://127.0.0.1:5000`.

3. **Upload an Image**:
   Upload a photo of a vehicle to see detected damages with confidence scores and bounding boxes.

---

## 📂 Project Structure

```text
├── scanner_app/           # Flask Web Application
│   ├── app.py             # Main backend logic
│   ├── static/            # CSS, JS, and uploaded images
│   └── templates/         # HTML templates (index.html)
├── dataset/               # Training data (Images & Labels)
├── runs/                  # Training outputs (Weights, Logs, Plots)
├── train_yolov8.py        # YOLOv8 Training Pipeline
├── train_rcnn_v2.py       # Faster R-CNN Training Pipeline
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

---

## 📊 Dataset: CarDD

This project uses the **CarDD Dataset**, a comprehensive dataset for car damage detection. The models are configured to detect:
- **0**: Dent
- **1**: Scratch
- **2**: Crack
- **3**: Shattered Glass
- **4**: Bumper Damage
- **5**: Deformation

---

## 🛡️ License
This project is for educational/research purposes in the field of Computer Vision and Vehicle Inspection.

> [!TIP]
> **GPU Optimization**: If you encounter Out-of-Memory (OOM) errors during training, try reducing the `batch` size in the training scripts. 
