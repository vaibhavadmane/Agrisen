from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
# Enable CORS for all routes (important for HTML/JS to communicate with the API)
CORS(app)

# Load the trained model
try:
    model = joblib.load('models/random_forest_crop_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("ERROR: Model 'models/random_forest_crop_model.pkl' not found. Please train it first.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data sent from the frontend
        data = request.get_json(force=True)

        # Extract features in the correct order
        features = [
            data['N'],
            data['P'],
            data['K'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]

        # Convert to numpy array and reshape for prediction
        input_data = np.array([features])

        # Make prediction
        prediction = model.predict(input_data)
        recommended_crop = prediction[0].upper()

        # Return the result as JSON
        return jsonify({
            'success': True,
            'recommended_crop': recommended_crop
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on port 5000
    print("Starting Flask API server on http://127.0.0.1:5000")
    app.run(port=5000)