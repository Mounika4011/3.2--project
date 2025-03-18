import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample Data (Replace this with actual dataset)
data = pd.DataFrame({
    "Age": [20, 21, 22, 23, 19, 24, 25, 26],
    "Gender": ["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"],
    "Course": ["Engineering", "Science", "Arts", "Engineering", "Science", "Arts", "Engineering", "Science"],
    "Year_of_Study": [3, 2, 1, 4, 3, 2, 1, 4],
    "GPA": [3.49, 3.1, 2.5, 3.8, 3.0, 2.9, 2.7, 3.6],
    "Married": ["No", "Yes", "No", "Yes", "No", "No", "Yes", "Yes"],
    "Anxiety": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    "Panic_Attacks": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    "Treatment": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
    "Depression": ["No", "Yes", "Yes", "No", "No", "Yes", "Yes", "No"]
})

# Encode categorical variables
encoder = LabelEncoder()
for col in ["Gender", "Course", "Married", "Anxiety", "Panic_Attacks", "Treatment", "Depression"]:
    data[col] = encoder.fit_transform(data[col])

# Define features (X) and target (y)
X = data.drop(columns=["Depression"])
y = data["Depression"]

# Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Ensure "model" directory exists
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save model and scaler
joblib.dump(model, os.path.join(model_dir, "logistic_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

# Function to Predict Depression for New Data
def predict_depression(new_data):
    df = pd.DataFrame([new_data])
    for col in ["Gender", "Course", "Married", "Anxiety", "Panic_Attacks", "Treatment"]:
        df[col] = encoder.transform(df[col])
    df = scaler.transform(df)
    prediction = model.predict(df)
    return "Yes" if prediction[0] == 1 else "No"

print("Model training and saving completed!")

