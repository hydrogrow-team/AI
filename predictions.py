import pickle

with open("DecisionTree.pkl",'rb') as f:
    DecisionTree = pickle.load(f)
with open("RandomForest.pkl",'rb') as f:
    Rf = pickle.load(f)

print(DecisionTree.predict([[113,22,79,7.388,90.422]]))

print(Rf.predict([[113,22,79,7.388,90.422]]))