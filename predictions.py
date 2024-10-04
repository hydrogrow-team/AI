import pickle

with open("DecisionTree.pkl",'rb') as f:
    DecisionTree = pickle.load(f)
with open("RandomForest.pkl",'rb') as f:
    Rf = pickle.load(f)

