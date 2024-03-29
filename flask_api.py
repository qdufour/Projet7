from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('./data/model.pkl')
scaler = joblib.load('./data/scaler.pkl')

@app.route('/')
def home_page():
    return 'Welcome to the credit scoring API'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        test_data_json = data['test_data']
        test_data_df = pd.read_json(test_data_json, orient='records')
        scaled_data = scaler.transform(test_data_df)
        prediction = model.predict_proba(scaled_data)[:, 1] 
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)