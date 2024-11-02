from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score

# load data from skleran
digits = load_digits()
x = digits.data
y = digits.target

model= RandomForestClassifier()

model.fit(x,y)

y_pred =model.predict(x)

AC =accuracy_score(y,y_pred)
print(f"Accuracy is {AC}")