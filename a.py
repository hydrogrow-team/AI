import pickle

with open("RandomForest.pkl", "rb") as f:
    model = pickle.load(f)

print(model)  