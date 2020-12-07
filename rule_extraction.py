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
import matplotlib.pyplot as plt


names = np.loadtxt("toy_decision2.data", max_rows=1, delimiter=",", dtype=str)
data = np.loadtxt("toy_decision2.data", skiprows=1, delimiter=",", dtype=str)

enc = OrdinalEncoder(dtype=np.float32)
data = enc.fit_transform(data)

mapping = {}

for variable_name, possible_values in zip(names, enc.categories_):
    for i, value_name in enumerate(possible_values):

        from_this = variable_name + " = " + str(float(i))
        to_that = variable_name + " = " + value_name
        # print(from_this, "==>", to_that)

        mapping[from_this] = to_that

y = data.T[0]
X = data.T[1:].T

def print_rules(pom_model):

    for i in range(len(model.states)):

        if isinstance(model.states[i].distribution, DiscreteDistribution):
            print(names[i], model.states[i].distribution.parameters)

        else:
            # Assume Categorical
            # print(names[i], model.structure[i], [names[j] for j in model.structure[i]])
            # print(np.array(model.states[i].distribution.parameters[0]))

            cpt = np.array(model.states[i].distribution.parameters[0])
            print("\n\n")

            for row in cpt:

                par_condition = "IF ("
                for j, par in enumerate([names[p] for p in model.structure[i]]):

                    seen = par + " = " + str(row[j])
                    if seen in mapping:
                        par_condition += mapping[seen]
                    else:
                        par_condition += seen

                    par_condition += " ^ "

                par_condition = par_condition[:-3]

                par_condition += ") THEN ("

                seen = names[i] + " = " + str(row[-2])
                if seen in mapping:
                    par_condition += mapping[seen]
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
    # tuple([3, 0]),
    tuple([4, 0]),
]

excluded = [
    tuple([0, 0]),
    tuple([0, 1]),
    tuple([0, 2]),
    tuple([0, 3]),
    tuple([0, 4]),
]

predictions = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    learning_data = np.c_[y_train, X_train]

    model = clf.from_samples(
        learning_data,
        algorithm='exact',
        exclude_edges=excluded,
        include_edges=required,
        state_names=[str(name) for name in names],
        max_parents=-1,
        # pseudocount=1.0,
    )
    print(model.structure)

    # print_rules(model)
    # import pdb; pdb.set_trace()

    nan_column = np.empty(y_test.shape)
    nan_column[:] = np.nan
    test_data = np.c_[nan_column, X_test]

    pred = model.predict_proba(test_data)

    # print(pred[0][0])

    predictions.append(
        [item[0].items()[1][1] > 0.5 for item in pred][0]
    )
    # print(predictions)

# import pdb; pdb.set_trace()

print(accuracy_score(np.array(predictions), y))
