from flask import Flask, request, jsonify
from joblib  import load
import numpy as np
from datetime  import datetime, timedelta
from flask_cors import CORS
from ultralytics import YOLO
import cv2

app = Flask(__name__)
CORS(app)
model_qty  = load('models/ensemble_qty.joblib')
model_date = load('models/ensemble_interval.joblib')

YOLO_MODEL_PATH = "models/best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

# 구매 주기 및 수량 예측
@app.route('/predict', methods=['POST'])
def predict():
    features = request.get_json().get('features', [])
    x = np.array(features).reshape(1, -1)

    qty = int(model_qty .predict(x)[0])
    days= int(model_date.predict(x)[0])
    next_date = (datetime.today() + timedelta(days=days)).date().isoformat()

    return jsonify({
      'recommended_qty': qty,
      'recommended_date': next_date
    })

# YOLO
@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    # 파일 → OpenCV 이미지
    file_bytes = request.files['image'].read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 추론
    results = yolo_model(img)[0]
    detections = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = box
        class_name = yolo_model.names[int(cls)]
        detections.append({
            'class_id':   int(cls),
            'class_name': class_name,
            'confidence': float(conf),
            'bbox':       [float(x1), float(y1), float(x2), float(y2)]
        })

    return jsonify({'detections': detections})

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)
