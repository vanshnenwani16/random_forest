from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the POST request
        features = request.json['features']
        
        # Convert to numpy array and reshape for single sample prediction
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get prediction probability
        probabilities = model.predict_proba(features)
        
        # Return prediction and probabilities
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': probabilities[0].tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
