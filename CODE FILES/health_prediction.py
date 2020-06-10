import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('.\heart.csv')

X = data[list(data.columns)[:-1]]
Y = data["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, Y_train)

pickle.dump(rf, open("./health_prediction_new_model.pkl","wb"))