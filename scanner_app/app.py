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

    # Run inference with a higher confidence threshold (0.50) to filter out noise
    results = model.predict(source=original_path, conf=0.50, save=False)
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

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        # Clean filename to remove any query strings like ?t=...
        filename = data.get('filename', '').split('?')[0]
        detections = data.get('detections', [])
        
        if not filename:
            return jsonify({'error': 'Filename missing'}), 400

        import io
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()

        # Title & Branding
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, fontSize=24, spaceAfter=20)
        elements.append(Paragraph("Damage Assessment Report", title_style))
        elements.append(Paragraph(f"Project: Detectra AI | File: {filename}", styles['Normal']))
        elements.append(Spacer(1, 20))

        # Images
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + filename)

        if os.path.exists(result_path):
            try:
                # Add image with fixed aspect ratio sizing
                img = Image(result_path, width=400, height=300, kind='proportional')
                elements.append(img)
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("<center>Annotated Detection View</center>", styles['Italic']))
            except Exception as e:
                elements.append(Paragraph(f"[Image could not be loaded: {str(e)}]", styles['Normal']))
        
        elements.append(Spacer(1, 30))

        # Table
        table_data = [["Damage Type", "Confidence"]]
        for det in detections:
            table_data.append([str(det['class']).capitalize(), f"{det['confidence']}%"])
        
        if len(table_data) == 1:
            table_data.append(["No damage detected", "-"])

        t = Table(table_data, colWidths=[200, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3b82f6")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(t)
        
        doc.build(elements)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'Damage_Report_{filename}.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"PDF Error: {str(e)}")
        return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
