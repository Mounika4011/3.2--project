from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model and scaler
model_path = "model/logistic_model.pkl"
scaler_path = "model/scaler.pkl"

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file not found. Please run `model.py` first.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Label encoding categories (must match training data)
categories = {
    "Gender": ["Male", "Female"],
    "Course": ["Engineering", "Science", "Arts"],
    "Married": ["Yes", "No"],
    "Anxiety": ["Yes", "No"],
    "Panic_Attacks": ["Yes", "No"],
    "Treatment": ["Yes", "No"],
}

# Function to process input and predict
def predict_depression(form_data):
    try:
        # Convert form inputs into a dictionary
        data = {
            "Age": int(form_data["Age"]),
            "Gender": form_data["Gender"],
            "Course": form_data["Course"],
            "Year_of_Study": int(form_data["Year_of_Study"]),
            "GPA": float(form_data["GPA"]),
            "Married": form_data["Married"],
            "Anxiety": form_data["Anxiety"],
            "Panic_Attacks": form_data["Panic_Attacks"],
            "Treatment": form_data["Treatment"]
        }

        # Validate categorical inputs
        for key in categories:
            if data[key] not in categories[key]:
                return "Invalid input! Please enter correct values."

        # Convert categorical values to numerical using index
        for key in categories:
            data[key] = categories[key].index(data[key])

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Scale numerical values
        df = scaler.transform(df)

        # Predict
        prediction = model.predict(df)
        return "Yes" if prediction[0] == 1 else "No"

    except Exception as e:
        return f"Error: {str(e)}"

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        prediction = predict_depression(request.form)
        return render_template("index.html", prediction=prediction)
    
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)


