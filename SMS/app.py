from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(port=5004, debug=True)