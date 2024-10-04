import flask,requests,pickle,jsonify

app = flask.Flask(__name__)

@app.route('/',methods=['GET'])
def send_data():
    data = requests.get("")
    with open("RandomFores.pkl",'rb') as f:
        model = pickle.load(f)
    predictions = model.predict_proba(data)
    return predictions


    pass


if __name__ == 'main':
    app.run()

    