from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the scaler
scaler = joblib.load('models/train_scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['input_data']
    input_data = pd.DataFrame([eval(input_data)])
    
    scaled_input_data = scaler.transform(input_data)
    
    model_choice = request.form['model']
    if model_choice == 'Naive Bayes':
        model = joblib.load('models/naive_bayes_model.joblib')
    elif model_choice == 'XGBoost':
        model = joblib.load('models/xgboost_model.joblib')
    
    prediction = model.predict(scaled_input_data)
    result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
