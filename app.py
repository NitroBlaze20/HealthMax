from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and symptoms list
rf_model = joblib.load("rf_model.pkl")  # Save and move your model here
symptoms_list = joblib.load("symptoms_list.pkl")  # Save and move your symptoms list here

@app.route("/")
def home():
    return render_template("index.html", symptoms_list=symptoms_list)

@app.route("/predict", methods=["POST"])
def predict():
    user_symptoms = request.form.getlist("symptoms")
    symptom_vector = {symptom: 0 for symptom in symptoms_list.values()}
    
    for symptom in user_symptoms:
        symptom_vector[symptom] = 1

    input_data = pd.DataFrame([symptom_vector])
    prediction = rf_model.predict(input_data)[0]

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
