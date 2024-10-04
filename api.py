from flask import Flask
import requests, pickle, json
from flask import request
app = Flask(__name__)


model = None
with open("RandomForest.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/predict", methods=["GET"])
def send_data():
    global model
    N = float(request.args.get("N"))
    temp = float(request.args.get("temp"))
    hum = float(request.args.get("hum"))
    ph = float(request.args.get("ph"))
    rainfall = float(request.args.get("rainfall"))
    data = [[N, temp, hum, ph, rainfall]]
    predictions = model.predict_proba(data)
    return json.dumps(predictions.tolist())


if __name__ == "__main__":
    app.run()
