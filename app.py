from flask import Flask, request, jsonify
import joblib
import pandas as pd
from xgboost import XGBClassifier

app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load('model/xgb_attrition_model.pkl')
feature_columns = joblib.load('model/feature_columns.pkl')  # Load expected columns

# Define a function to preprocess incoming data
def preprocess(data):
    # Apply the same preprocessing as during training
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})

    # One-hot encoding for categorical variables
    data = data.join(pd.get_dummies(data['BusinessTravel'], prefix='BusinessTravel')).drop('BusinessTravel', axis=1)
    data = data.join(pd.get_dummies(data['Department'], prefix='Department')).drop('Department', axis=1)
    data = data.join(pd.get_dummies(data['EducationField'], prefix='EducationField')).drop('EducationField', axis=1)
    data = data.join(pd.get_dummies(data['JobRole'], prefix='JobRole')).drop('JobRole', axis=1)
    data = data.join(pd.get_dummies(data['MaritalStatus'], prefix='MaritalStatus')).drop('MaritalStatus', axis=1)

    # Replace True/False with 1/0 and handle future downcasting warnings
    data.replace({True: 1, False: 0}, inplace=True)
    data = data.infer_objects(copy=False)

    # Align input data with the saved feature columns (add missing columns with zeros)
    for col in feature_columns:
        if col not in data.columns:
            data[col] = 0

    # Ensure the columns are in the same order as the training set
    data = data[feature_columns]

    return data

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Convert JSON into a DataFrame
        df = pd.DataFrame([data])

        # Preprocess the data
        df_preprocessed = preprocess(df)

        # Make prediction
        prediction_proba = model.predict_proba(df_preprocessed)[:, 1]

        # Apply threshold to decide between 0 or 1
        threshold = 0.35
        prediction = (prediction_proba >= threshold).astype(int)

        # Return the prediction as JSON
        result = {"Attrition Prediction": int(prediction[0]), "Prediction Probability": float(prediction_proba[0])}
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)