import os
import pandas as pd
import json
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
port = int(os.getenv("PORT", 1010))

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        try:
            post_data = request.get_json()
            print(post_data)
            json_data = json.dumps(post_data)
            print(json_data)
            input_data = pd.read_json(json_data)
            print(input_data)
            model = pickle.load(open("health_prediction_model_PK.pkl", "rb"))
            prediction_value = model.predict(input_data)
            print(prediction_value)
            return str(prediction_value)
        except Exception as e:
            return (e)
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)