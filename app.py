from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import numpy as np

app = Flask(__name__)
CORS(app)

model = load('purchase_model.joblib')

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get('features', [])
    x = np.array(features).reshape(1, -1)
    pred = model.predict(x)[0]
    return jsonify({"recommended_qty": int(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)