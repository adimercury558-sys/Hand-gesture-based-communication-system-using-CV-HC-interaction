#Deeraj_47_github @HAKAISHIN_KAGGLE
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

data = pd.read_csv("sign_data.csv")
X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

with open("sign_model.pkl", "wb") as f:
    pickle.dump(model, f)
