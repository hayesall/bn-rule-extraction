# Copyright © 2020 Alexander L. Hayes

"""
Extracting decision rules from Bayesian Networks
"""

from pomegranate import BayesianNetwork
from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import numpy as np


names = np.loadtxt("toy_decision.csv", max_rows=1, delimiter=",", dtype=str)
data = np.loadtxt("toy_decision.csv", skiprows=1, delimiter=",", dtype=str)

enc = OrdinalEncoder(dtype=np.float32)
data = enc.fit_transform(data)
print(enc.categories_)

# TODO(hayesall): ``mapping`` is basically a "pretty-printer", this could
#       probably be included as part of the ``print_rules`` function.
mapping = {}
for variable_name, possible_values in zip(names, enc.categories_):
    for i, value_name in enumerate(possible_values):

        from_this = variable_name + " = " + str(float(i))
        to_that = variable_name + " = " + value_name
        mapping[from_this] = to_that

y = data.T[0]
X = data.T[1:].T

def print_rules(pom_model, variable_mapping):

    for i in range(len(model.states)):

        if isinstance(model.states[i].distribution, DiscreteDistribution):
            print(names[i], model.states[i].distribution.parameters)

        else:
            # Assume isinstance Categorical

            cpt = np.array(model.states[i].distribution.parameters[0])
            print("\n\n")

            for row in cpt:

                par_condition = "IF ("
                for j, par in enumerate([names[p] for p in model.structure[i]]):

                    seen = par + " = " + str(row[j])
                    if seen in variable_mapping:
                        par_condition += variable_mapping[seen]
                    else:
                        par_condition += seen

                    par_condition += " ^ "

                par_condition = par_condition[:-3]

                par_condition += ") THEN ("

                seen = names[i] + " = " + str(row[-2])
                if seen in variable_mapping:
                    par_condition += variable_mapping[seen]
                else:
                    par_condition += seen

                par_condition += ")"

                _conf_factor = row[-1] / (1 - row[-1])

                if _conf_factor >= 1.0:
                    print(par_condition)
                    print("\tCF = {0:0.2f}".format(_conf_factor))


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
        state_names=[str(name) for name in names],
        max_parents=-1,
    )
    print(model.structure)

    if test_index == 0:
        print("Decision rules extracted from the first test:\n")
        print_rules(model, mapping)

    nan_column = np.empty(y_test.shape)
    nan_column[:] = np.nan
    test_data = np.c_[nan_column, X_test]

    pred = model.predict_proba(test_data)

    predictions.append(
        [item[0].items()[1][1] > 0.5 for item in pred][0]
    )

print(accuracy_score(np.array(predictions), y))
