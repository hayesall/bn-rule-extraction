# Copyright © 2021 Alexander L. Hayes

"""
Extracting decision rules from Bayesian Networks
"""

from bayes_rule_extraction import print_rules, ordinal_encode
from pomegranate import BayesianNetwork
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

data = pd.read_csv("toy_decision.csv")
names = data.columns
encoded, mapping = ordinal_encode(names, data)

X = encoded[:, 1:]
y = encoded[:, 0]

loo = LeaveOneOut()
clf = BayesianNetwork()

required = [
    tuple([1, 0]),
    tuple([4, 0]),
]

predictions = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    learning_data = np.c_[y_train, X_train]

    model = clf.from_samples(
        learning_data,
        algorithm='exact',
        include_edges=required,
        state_names=names,
        max_parents=-1,
    )

    if test_index == 0:
        print("Decision rules extracted from the first test:\n")
        print_rules(model, names, mapping)

    nan_column = np.empty(y_test.shape)
    nan_column[:] = np.nan
    test_data = np.c_[nan_column, X_test]

    pred = model.predict_proba(test_data)

    predictions.append(
        [item[0].items()[1][1] > 0.5 for item in pred][0]
    )

print(accuracy_score(np.asarray(predictions), y))
