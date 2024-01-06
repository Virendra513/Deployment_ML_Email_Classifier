from flask import Flask, render_template, request
import pickle
import joblib

loaded_objects = joblib.load("models/combined_objects.pkl")
model = loaded_objects['model']
tokenizer = loaded_objects['vectorizer']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])

def predict():
    if request.method=="POST":
        email= request.form.get("content")
    tokenzied_email=tokenizer.transform([email])
    prediction=model.predict(tokenzied_email)
    prediction= 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email )

if __name__ == '__main__':
    app.run(debug=True)