import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).parent
app = Flask(__name__, 
            static_folder=str(BASE_DIR / "static"),
            template_folder=str(BASE_DIR / "templates"))

# Config
UPLOAD_FOLDER = os.path.join(str(BASE_DIR), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to YOLOv8 weights
MODEL_PATH = os.path.join(str(BASE_DIR), 'models', 'best.pt')

# Load model once at startup
print(f"Loading YOLOv8 model from {MODEL_PATH}...")
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save original image
    filename = file.filename
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'original_' + filename)
    file.save(original_path)

    # Run inference
    results = model.predict(source=original_path, conf=0.25, save=False)
    res = results[0]
    
    # Save annotated image
    annotated_img = res.plot()
    result_filename = 'detected_' + filename
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, annotated_img)

    # Extract detection info
    detections = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]
        detections.append({
            'class': name,
            'confidence': round(conf * 100, 1)
        })

    return jsonify({
        'original_url': f'/static/uploads/original_{filename}',
        'result_url': f'/static/uploads/detected_{filename}',
        'detections': detections
    })

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
