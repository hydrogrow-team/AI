from flask import Flask
import  pickle, json
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
    predictions = predictions[0]
    val = []
    for class_, prob in zip(model.classes_, predictions):
        val.append(( class_,prob))
    val.sort(key=lambda x: x[1], reverse=True)
    top_3 = []
    for class_, prob in val[:3]:
        if prob:
            top_3.append((class_, prob))    
        else:
            break
    

    return json.dumps(top_3)


if __name__ == "__main__":
    app.run()
