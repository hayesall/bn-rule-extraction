# Copyright © 2021 Alexander L. Hayes

"""
Rule extraction from Bayesian Networks
"""

from pomegranate import DiscreteDistribution
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


def ordinal_encode(names, data):
    encoder = OrdinalEncoder(dtype=np.float32)
    encoded = encoder.fit_transform(data)
    mapping = {}
    for variable_name, possible_values in zip(names, encoder.categories_):
        for i, value_name in enumerate(possible_values):

            from_this = variable_name + " = " + str(float(i))
            to_that = variable_name + " = " + value_name
            mapping[from_this] = to_that

    return encoded, mapping


def print_rules(pom_model, variable_names, variable_mapping):

    unconditioned_rules = []
    conditioned_rules = []

    for i in range(len(pom_model.states)):

        if isinstance(pom_model.states[i].distribution, DiscreteDistribution):
            unconditioned_rules.append((variable_names[i], pom_model.states[i].distribution.parameters[0]))

        else:
            # Assume isinstance Categorical

            cpt = np.asarray(pom_model.states[i].distribution.parameters[0])

            for row in cpt:

                par_condition = "IF ("
                for j, par in enumerate([variable_names[p] for p in pom_model.structure[i]]):

                    seen = par + " = " + str(row[j])
                    if seen in variable_mapping:
                        par_condition += variable_mapping[seen]
                    else:
                        par_condition += seen

                    par_condition += " ^ "

                par_condition = par_condition[:-3]

                par_condition += ") THEN ("

                seen = variable_names[i] + " = " + str(row[-2])
                if seen in variable_mapping:
                    par_condition += variable_mapping[seen]
                else:
                    par_condition += seen

                par_condition += ")"

                _conf_factor = row[-1] / (1 - row[-1])

                if _conf_factor >= 1.0:
                    par_condition += "\n\tCF = {0:0.2f}".format(_conf_factor)
                    conditioned_rules.append(par_condition)

    print("Probabilities:")
    for rule in unconditioned_rules:
        print("-", rule[0])
        for value in rule[1]:

            temp_val = variable_mapping[rule[0] + " = " + str(value)]

            print(
                "  P(",
                f"{temp_val}",
                ") = {0:0.2f}".format(rule[1][value])
            )

    print()
    for rule in conditioned_rules:
        print(rule)
