# Rule Extraction from Bayesian Networks

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
- Outlook       {'overcast': 0.31, 'rain': 0.38, 'sunny': 0.31}
- Temperature   {'cool': 0.30, 'hot': 0.23, 'mild': 0.47}
- Wind          {'strong': 0.46, 'weak': 0.54}


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
	CF = 1.00
IF (Outlook = sunny ^ Wind = weak) THEN (PlayTennis = yes)
	CF = 1.00


IF (Temperature = cool) THEN (Humidity = normal)
	CF = inf
IF (Temperature = hot) THEN (Humidity = high)
	CF = 2.00
IF (Temperature = mild) THEN (Humidity = high)
	CF = 2.00
```

## Getting Started

Install requirements:

```console
pip install -r requirements.txt
```

Run `rule_extraction.py`. This shows rules and LOOCV accuracy as a metric.

```console
python rule_extraction.py
```

A `scikit-learn` decision tree is included for comparison
(C4.5 was a baseline in the paper). This prints LOOCV accuracy as a metric.

```console
python decision_tree.py
```

## Notes

- This is currently built for personal use and early experimentation, some major
  cleanup is needed before production use or using it as a baseline in your work.
- *Gopalakrishnan 2010* used a modified version of K2 for structure learning,
  *this uses* `pomegranate.BayesianNetwork.from_samples` method for fitting
  exact structures.
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
