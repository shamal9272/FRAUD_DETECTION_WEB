from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]

    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "🚨 Spam / Fraud Message"
    else:
        result = "✅ Genuine Message"

    return render_template("index.html", prediction=result)

app.run(host='0.0.0.0', port=5000, debug=True)