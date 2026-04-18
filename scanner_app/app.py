import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).parent
app = Flask(__name__, 
            static_folder=str(BASE_DIR / "static"),
            template_folder=str(BASE_DIR / "templates"))

# Config
UPLOAD_FOLDER = os.path.join(str(BASE_DIR), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model Paths
YOLO_PATH = os.path.join(str(BASE_DIR), 'models', 'best.pt')
RCNN_PATH = os.path.join(str(BASE_DIR), 'models', 'faster_rcnn.pt')
CLASS_NAMES = ["dent", "scratch", "crack", "shattered_glass", "bumper_damage", "deformation"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global Model Instances
models = {'yolo': None, 'rcnn': None}

def load_models():
    # Load YOLOv8
    if os.path.exists(YOLO_PATH):
        print(f"Loading YOLOv8 from {YOLO_PATH}...")
        models['yolo'] = YOLO(YOLO_PATH)
    
    # Load Faster R-CNN
    if os.path.exists(RCNN_PATH):
        print(f"Loading Faster R-CNN from {RCNN_PATH}...")
        rcnn = fasterrcnn_resnet50_fpn(weights=None)
        in_features = rcnn.roi_heads.box_predictor.cls_score.in_features
        rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7) # 6 classes + bg
        
        # Load state dict
        ckpt = torch.load(RCNN_PATH, map_location=DEVICE)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        rcnn.load_state_dict(state_dict)
        rcnn.to(DEVICE)
        rcnn.eval()
        models['rcnn'] = rcnn

load_models()

def predict_rcnn(img_path):
    img = Image.open(img_path).convert("RGB")
    original_size = img.size # (W, H)
    
    # Preprocess
    img_t = torchvision.transforms.functional.to_tensor(img).to(DEVICE)
    
    with torch.no_grad():
        prediction = models['rcnn']([img_t])[0]
    
    # Filter detections
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    keep = scores >= 0.25
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    # Map to Detectra standard format
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        name = CLASS_NAMES[int(label)-1]
        detections.append({
            'class': name,
            'confidence': round(float(score) * 100, 1),
            'box': box.tolist() # [x1, y1, x2, y2]
        })
    
    # Draw annotations
    cv_img = cv2.imread(img_path)
    for det in detections:
        x1, y1, x2, y2 = map(int, det['box'])
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(cv_img, f"{det['class']} {det['confidence']}%", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return cv_img, detections

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model', 'yolo')
    
    if models.get(model_choice) is None:
        return jsonify({'error': f'Model {model_choice} not loaded on server'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = file.filename
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
    file.save(original_path)
    
    detections = []
    result_filename = 'detected_' + filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

    if model_choice == 'yolo':
        results = models['yolo'].predict(source=original_path, conf=0.25, save=False)
        res = results[0]
        annotated_img = res.plot()
        cv2.imwrite(result_path, annotated_img)
        
        for box in res.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = models['yolo'].names[cls_id]
            detections.append({'class': name, 'confidence': round(conf * 100, 1)})
            
    else: # rcnn
        annotated_img, detections = predict_rcnn(original_path)
        cv2.imwrite(result_path, annotated_img)

    return jsonify({
        'original_url': f'/static/uploads/original_{filename}',
        'result_url': f'/static/uploads/detected_{filename}',
        'detections': detections,
        'model_used': model_choice
    })

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
