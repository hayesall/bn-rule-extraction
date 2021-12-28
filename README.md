# Extracting Interpretable Rules from Bayesian Networks

Based on the 2010 paper: "*Bayesian rule learning for biomedical data mining*"
by Vanathi Gopalakrishnan, Jonathan L. Lustgarten, Shyam Visweswaran, and
Gregory F. Cooper.

## Overview

Given a Bayesian Network structure and parameters in the form of conditional
probability tables, Gopalakrishnan et al. proposed to extract if/then rules
based on the edges and the ratio between outcome cases
(from Fig. 3: "CF is expressed as the
likelihood ratio of the conditional probability of the target value given
the value of its parent variables.")

For example, this structure (CPTs not shown):

![Structure of a Bayesian Network learned on the toy play tennis data set. This shows that whether a person plays tennis is dependent on the weather outlook and the wind; and suggests that temperature affects humidity, but both are independent of whether someone plays tennis.](docs/play_tennis_bn_1.png)

Can be turned into the following rules:

```text
Probabilities:
- Outlook
  P( Outlook = sunny ) = 0.36
  P( Outlook = overcast ) = 0.29
  P( Outlook = rain ) = 0.36
- Temperature
  P( Temperature = hot ) = 0.29
  P( Temperature = mild ) = 0.43
  P( Temperature = cool ) = 0.29
- Wind
  P( Wind = weak ) = 0.57
  P( Wind = strong ) = 0.43

IF (Outlook = overcast ^ Wind = strong) THEN (PlayTennis = yes)
	CF = inf
IF (Outlook = overcast ^ Wind = weak) THEN (PlayTennis = yes)
	CF = inf
IF (Outlook = rain ^ Wind = strong) THEN (PlayTennis = no)
	CF = inf
IF (Outlook = rain ^ Wind = weak) THEN (PlayTennis = yes)
	CF = inf
IF (Outlook = sunny ^ Wind = strong) THEN (PlayTennis = no)
	CF = 1.00
IF (Outlook = sunny ^ Wind = strong) THEN (PlayTennis = yes)
	CF = 1.00
IF (Outlook = sunny ^ Wind = weak) THEN (PlayTennis = no)
	CF = 2.00
IF (Temperature = cool) THEN (Humidity = normal)
	CF = inf
IF (Temperature = hot) THEN (Humidity = high)
	CF = 3.00
IF (Temperature = mild) THEN (Humidity = high)
	CF = 2.00
```

## Getting Started

### Notebook Demos

Some demos are implemented as Jupyter notebooks:

| Notebook | Colab Link | View on GitHub |
| :---- | :---- | ----: |
| Mitchell Tennis Dataset | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hayesall/bn-rule-extraction/blob/main/docs/notebooks/tennis.ipynb) | [`tennis.ipynb`](https://github.com/hayesall/bn-rule-extraction/blob/main/docs/notebooks/tennis.ipynb) |
| Adult Dataset | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hayesall/bn-rule-extraction/blob/main/docs/notebooks/adult.ipynb) | [`adult.ipynb`](https://github.com/hayesall/bn-rule-extraction/blob/main/docs/notebooks/adult.ipynb) |

### Working with the Python package

Clone + install requirements:

```console
git clone https://github.com/hayesall/bn-rule-extraction.git
cd bn-rule-extraction
pip install -e .
```

The `bayes_rule_extraction` package exposes two functions: `print_rules` and `ordinal_encode`.

Here's a minimal working example:

```python
from bayes_rule_extraction import ordinal_encode, print_rules
from pomegranate import BayesianNetwork
import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/hayesall/bn-rule-extraction/main/toy_decision.csv")

encoded, mapping = ordinal_encode(df.columns, df)

# Encode a constraint that "PlayTennis" cannot be the parent of any other node.
excluded_edges = [tuple([0, i]) for i in range(1, len(df.columns))]

model = BayesianNetwork().from_samples(
    encoded,
    exclude_edges=excluded_edges,
    state_names=df.columns,
)

print_rules(model, df.columns, mapping)
```

## Notes

- This is implemented as an "*explanation method*" to help explain a Bayesian Network.
  It's not currently possible to use the extracted rules directly for classification.
- *Gopalakrishnan 2010* used a modified version of K2 for structure learning.
- The `include_edges` parameter in the `pomegranate.BayesianNetwork.from_samples`
  method seems to be required to learn "*interesting*" or "*useful*" rules,
  especially if there is a specific outcome variable (like `PlayTennis`)
  you are interested in. This might be explained by differences in structure
  learning methods&mdash;variable ordering in K2 provides some control
  over influence between possible parents and children.

## Acknowledgements

The Toy Decision data set is lifted from Tom Mitchell's *Machine Learning* book,
see section 3.4.2 (page 59 in my edition).

### BibTex

```bibtex
@article{gopalakrishnan2010bayesian,
  author = {Gopalakrishnan, Vanathi and Lustgarten, Jonathan L. and Visweswaran, Shyam and Cooper, Gregory F.},
  title = "{Bayesian rule learning for biomedical data mining}",
  journal = {Bioinformatics},
  volume = {26},
  number = {5},
  pages = {668-675},
  year = {2010},
  month = {01},
  issn = {1367-4803},
  doi = {10.1093/bioinformatics/btq005},
  url = {https://doi.org/10.1093/bioinformatics/btq005},
  eprint = {https://academic.oup.com/bioinformatics/article-pdf/26/5/668/16897540/btq005.pdf},
}
```
