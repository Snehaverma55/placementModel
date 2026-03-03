from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    cgpa = float(request.form["cgpa"])
    internship = int(request.form["internship"])
    coding_rating = int(request.form["coding_rating"])
    projects = int(request.form["projects"])

    input_data = pd.DataFrame([[cgpa, internship, coding_rating, projects]],
                              columns=["CGPA", "Internship", "Coding_Rating", "Projects"])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    result = "Placed ✅" if prediction[0] == 1 else "Not Placed ❌"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)