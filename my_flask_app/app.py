from flask import Flask, request, jsonify
from joblib  import load
import numpy as np
from datetime  import datetime, timedelta
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model_qty  = load('models/ensemble_qty.joblib')
model_date = load('models/ensemble_interval.joblib')

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

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)
