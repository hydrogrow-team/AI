from flask import Flask
import requests, pickle, json

app = Flask(__name__)


model = None
with open("RandomFores.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["GET"])
def send_data():
    global model
    data = requests.get("")
    predictions = model.predict_proba(data)
    return predictions


if __name__ == "__main__":
    app.run(debug=True)
