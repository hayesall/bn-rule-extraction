# Copyright © 2020 Alexander L. Hayes

"""
Decision Tree Classifier for demonstration

TODO:

- Extract the decision rules in a similar format for comparison.
    This could serve as some inspiration:
    https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree/39772170#39772170
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np


names = np.loadtxt("toy_decision.csv", max_rows=1, delimiter=",", dtype=str)
data = np.loadtxt("toy_decision.csv", skiprows=1, delimiter=",", dtype=str)

enc = OrdinalEncoder(dtype=np.float32)
data = enc.fit_transform(data)

y = data.T[0]
X = data.T[1:].T

loo = LeaveOneOut()

clf = DecisionTreeClassifier(
    max_depth=3,
    random_state=0,
)

predictions = []

# TODO(hayesall): cross_val_predict would be cleaner, but is harder to incorporate with Bayes net currently.
#   A ``GopalakrishnanDecisionTree`` object might be cleaner.
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    predictions.append(pred[0])

print(accuracy_score(np.array(predictions), y))
