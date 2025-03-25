# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from preprocess import preprocessing_data

app = Flask(__name__)

# Load the model, preprocessor and LabelEncoder
model_data = joblib.load('model_pipeline.pkl')
model = model_data['model']
label_encoder = model_data['label_encoder']
preprocessor = model_data['preprocessor']

@app.route('/')
def home():
    return "Welcome to the model prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)
    
    # Convert the JSON data to a DataFrame
    input_data = pd.DataFrame([data])

    input_data = input_data.fillna(value=np.nan)

    #print("Input data: ", input_data)

    # Preprocess the input data
    input_data = preprocessing_data(input_data)

    input_data = preprocessor.transform(input_data)


    # Predictions
    predictions = model.predict(input_data)[:]
    
    # Decode the predictions to original labels
    decoded_predictions = label_encoder.inverse_transform(predictions)
    
    # Convert predictions to a list
    decoded_predictions = decoded_predictions.tolist()
    
    # Return the predictions as a JSON response
    return jsonify(predictions=decoded_predictions)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
