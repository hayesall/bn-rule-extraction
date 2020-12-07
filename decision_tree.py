# Copyright © 2020 Alexander L. Hayes

"""
Decision Tree Classifier for demonstration
"""

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import plot_tree

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt


names = np.loadtxt("toy_decision.data", max_rows=1, delimiter=",", dtype=str)
data = np.loadtxt("toy_decision.data", skiprows=1, delimiter=",", dtype=str)

enc = OrdinalEncoder(dtype=np.float32)
data = enc.fit_transform(data)

y = data.T[0]
X = data.T[1:].T

###
# https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree/39772170#39772170
from sklearn.tree import _tree

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
###

loo = LeaveOneOut()

clf = DecisionTreeClassifier(
    max_depth=3,
    random_state=0,
)

'''
_predictions = cross_val_predict(
    clf,
    X=X,
    y=y,
    cv=loo,
)
print(accuracy_score(_predictions, y))
'''

predictions = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)

    if test_index == 0:
        # tree_to_code(clf, names[1:])

        fig = plt.figure(figsize=(20,15))
        _ = plot_tree(
            clf,
            feature_names=list(names[1:]),
            class_names=["No Tennis", "Tennis"],
            filled=True,
            label=None,
            impurity=False,
        )
        # plt.show()

    pred = clf.predict(X_test)
    predictions.append(pred[0])

print(accuracy_score(np.array(predictions), y))
