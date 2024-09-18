import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # To handle the imbalance

# Load your dataset (replace this with your actual dataset)
df = pd.read_csv('data\WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Step 1: Data Preprocessing
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Over18'] = df['Over18'].map({'Y': 1, 'N': 0})
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# One-hot encoding for categorical features
df = df.join(pd.get_dummies(df['BusinessTravel'], prefix='BusinessTravel')).drop('BusinessTravel', axis=1)
df = df.join(pd.get_dummies(df['Department'], prefix='Department')).drop('Department', axis=1)
df = df.join(pd.get_dummies(df['EducationField'], prefix='EducationField')).drop('EducationField', axis=1)
df = df.join(pd.get_dummies(df['JobRole'], prefix='JobRole')).drop('JobRole', axis=1)
df = df.join(pd.get_dummies(df['MaritalStatus'], prefix='MaritalStatus')).drop('MaritalStatus', axis=1)

# Convert True/False to 1/0
df.replace({True: 1, False: 0}, inplace=True)

# Drop unnecessary columns
df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)

# Step 2: Split the data into training and testing sets
X, y = df.drop('Attrition', axis=1), df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Handle the imbalance using SMOTE
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Step 4: Train the model (XGBoost in this case, replace with RandomForest or any model if needed)
model = XGBClassifier(scale_pos_weight=len(y_train_resampled[y_train_resampled == 0]) / len(y_train_resampled[y_train_resampled == 1]),
                      random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 6: Save the trained model and feature columns
joblib.dump(model, 'model/xgb_attrition_model.pkl')
joblib.dump(X_train_resampled.columns.tolist(), 'model/feature_columns.pkl')  # Save the feature columns

print("Model trained and saved successfully!")