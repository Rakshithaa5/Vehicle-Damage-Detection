---
title: Detectra AI | Vehicle Damage Scanner
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
---

# 🏎️ Detectra AI: Vehicle Damage Scanner

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/rakshithaa5/Vehicle_Damage_Detection)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8m-blueviolet?style=for-the-badge)](https://ultralytics.com/)
[![INR](https://img.shields.io/badge/Currency-INR-green?style=for-the-badge)](https://en.wikipedia.org/wiki/Indian_rupee)

**Detectra AI** is a professional-grade computer vision platform designed for automated vehicle inspection. It leverages high-precision **YOLOv8** object detection to identify damage and provides an instant **INR (₹) Repair Cost Estimate** for workshops and insurance agencies.

---

## ✨ Key Features

- 💰 **INR Repair Cost Estimator**: Automated financial assessment based on Indian market standards for dents, scratches, and structural damage.
- 📸 **Instant Camera Capture**: Mobile-optimized feature allowing users to photograph vehicles in real-time directly from their browser.
- ↕️ **Interactive Comparison Slider**: Premium UI component to compare original car photos with AI detected layers side-by-side.
- 📊 **Real-time Session Analytics**: Track scan counts and total damage findings during an inspection session.
- 🎯 **Optimized Precision**: Tuned detection engine (Threshold: 0.50) to minimize false positives in complex environments.

---

## 🚀 Live Access

The project is hosted and ready for demonstration:
👉 **[Detectra AI Live Demo](https://huggingface.co/spaces/rakshithaa5/Vehicle_Damage_Detection)**

---

## 📂 Project Structure

```text
├── scanner_app/           # Flask Web Application
│   ├── app.py             # Backend API & Server logic
│   ├── static/            # CSS (Glassmorphism), JS (Camera/Slider), Uploads
│   └── templates/         # UI Architecture (index.html)
├── models/                # Production Weights (best.pt)
├── train_yolov8.py        # High-performance Training Pipeline
├── requirements.txt       # Optimized Environment
└── README.md              # Project Documentation
```

---

## 🛠️ Local Installation

### 1. Clone & Setup
```bash
git clone https://github.com/Rakshithaa5/Vehicle-Damage-Detection
cd dl
python -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
python scanner_app/app.py
```
*Navigate to `http://localhost:5000` to access the scanner.*

---

## 📊 Dataset & Categories (CarDD)

The engine is trained on the comprehensive **CarDD** dataset, specialized in:
- **Dent**: ₹4,000 - ₹7,000
- **Scratch**: ₹1,500 - ₹3,000
- **Crack**: ₹2,500 - ₹5,000
- **Glass / Lamp Damage**: ₹4,000 - ₹12,000
- **Deformation**: ₹10,000+

---

## 🛡️ License
Distributed under the Apache-2.0 License. Built for Computer Vision research and professional portfolio demonstration.

---
> [!IMPORTANT]
> **Mobile Usage**: For the best experience with **Instant Camera Capture**, open the live link on a mobile device and grant camera permissions when prompted.
