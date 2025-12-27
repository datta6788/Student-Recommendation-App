from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# -----------------------------
# LOAD DATA & MODEL
# -----------------------------
df = pd.read_csv("data/students.csv")

model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# ROUTES
# -----------------------------

@app.route("/", methods=["GET"])
def login():
    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    return render_template("dashboard.html")


# 🔹 GET ALL STUDENT NAMES
@app.route("/students", methods=["GET"])
def students():
    return jsonify(df["name"].tolist())


# 🔹 GET SINGLE STUDENT DATA
@app.route("/student/<name>", methods=["GET"])
def get_student(name):
    student = df[df["name"] == name]

    if student.empty:
        return jsonify({"error": "Student not found"}), 404

    row = student.iloc[0]

    return jsonify({
        "maths": int(row["maths"]),
        "science": int(row["science"]),
        "english": int(row["english"]),
        "attendance": float(row["attendance"])
    })


# 🔹 ML PREDICTION
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([{
        "maths": data["maths"],
        "science": data["science"],
        "english": data["english"],
        "attendance": data["attendance"]
    }])

    scaled = scaler.transform(input_df)
    prediction = int(model.predict(scaled)[0])
    probability = model.predict_proba(scaled)[0][1]

    return jsonify({
        "prediction": prediction,
        "risk_probability": round(probability * 100, 2)
    })


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
