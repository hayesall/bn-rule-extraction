from setuptools import setup
from setuptools import find_packages
from codecs import open
from os import path

_here = path.abspath(path.dirname(__file__))
with open(path.join(_here, "README.md"), "r", encoding="utf-8") as _fh:
    LONG_DESCRIPTION = _fh.read()

setup(
    name="bayes-rule-extraction",
    packages=find_packages(exclude=["tests"]),
    package_dir={"bayes_rule_extraction": "bayes_rule_extraction"},
    author="Alexander L. Hayes",
    author_email="alexander@batflyer.net",
    description="Rule extraction from Bayesian Networks",
    version="0.1.0",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT License",
    python_requires=">=3.7",
    install_requires=["numpy>=1.19.0", "pomegranate>=0.14.0"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
