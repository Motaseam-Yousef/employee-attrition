# Employee Attrition Prediction API

This project is a Flask-based API that uses an XGBoost model to predict employee attrition. The model is trained using a dataset that includes various employee attributes. The API allows users to input employee data and receive a prediction on whether the employee is likely to leave the organization.

## Features

- **Preprocessing**: Categorical features like gender, overtime, and job roles are encoded for model training.
- **SMOTE**: Handles imbalanced datasets by using Synthetic Minority Oversampling Technique (SMOTE).
- **Model**: XGBoost is used as the predictive model.
- **API**: Flask-based API that takes employee data as JSON and returns an attrition prediction.

## Prerequisites

- Python 3.7+
- pip

### Libraries

Install the following Python libraries:

```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### 2. Train the Model

To train the model, run the following script:

```bash
python train_model.py
```

This script will:
- Preprocess the dataset.
- Handle class imbalance with SMOTE.
- Train an XGBoost classifier.
- Save the trained model and feature columns to the `model/` directory.

### 3. Start the Flask API

Run the Flask app:

```bash
python app.py
```

The app will start on `http://127.0.0.1:5000/`.

### 4. Make Predictions

You can use Postman, `curl`, or any other HTTP client to send a `POST` request to the `/predict` endpoint.

Example using `curl`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "Gender": "Male",
    "OverTime": "Yes",
    "BusinessTravel": "Travel_Rarely",
    "Department": "Sales",
    "EducationField": "Life Sciences",
    "JobRole": "Sales Executive",
    "MaritalStatus": "Single"
}'
```

### 5. Example JSON Input

The API expects the following format in the `POST` request:

```json
{
    "Gender": "Male",
    "OverTime": "Yes",
    "BusinessTravel": "Travel_Rarely",
    "Department": "Sales",
    "EducationField": "Life Sciences",
    "JobRole": "Sales Executive",
    "MaritalStatus": "Single"
}
```

### 6. Example API Response

```json
{
  "Attrition Prediction": 0,
  "Prediction Probability": 0.15
}
```

## Files

- **train_model.py**: Script to preprocess the dataset, handle imbalance, and train the XGBoost model.
- **app.py**: Flask app to expose the model as an API.
- **model/xgb_attrition_model.pkl**: Saved trained model.
- **model/feature_columns.pkl**: List of features used during training to ensure consistency during prediction.

## Dataset

This project assumes you have a dataset of employees that contains the following features:

- `Gender`, `OverTime`, `BusinessTravel`, `Department`, `EducationField`, `JobRole`, `MaritalStatus`, and others.

You can customize the dataset and preprocessing steps in `train_model.py` as needed.

## License

This project is licensed under the MIT License.