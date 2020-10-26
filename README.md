# ML-Project1
Project 1 of the ML course @ EPFL (CS-433)


## summary 
This repo contains the machine learning system to tackle the Higgs Boson Dataset using regression techniques. THe models use only numpy and matplotlib for visualisation. 

## dataset
Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014 - see [here](http://opendata.cern.ch/record/328)

## Project Structure
- `/EDA` contains notebooks for data exploration and feature engineering exploration
- `/dataset` contains the original dataset and our test and train set. 
- `/helpers` contains helper functions for the IO and for the implementations
- `/models` contains the notebooks for model and feature set exploration
- `/test`contains unittests for the implementations and the helpers
- `implementations.py` contains the implementations of the 6 machine learning algorithms and some extension/additions to these algorithms
- `preprocessing.py` contains the preprocessing pipelines that we use on the raw dataset before training
- `run.py` is a script that reproduces our top model predictions, it creates a `submission.csv` file that contains the predictions
- `plotting_regression_*.py` are scripts for plotting the behaviour of the logistic regression for different learning rates, these were used to gain insight into the influence of the learning rates

## Developer information
### Coding Style
Documentation according to the **docstring - reST** style 

see https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html for a small example


### Testing
Testing is done using [Unittest](https://docs.python.org/3/library/unittest.html), which is the built-int test framework for python. 

Tests are grouped in the `/test` folder. For tests to be auto-discoverable by the framework the files need to start with `test` and the tests need to be in a class that inherits from the `unittest.TestCase` baseclass. 

You can then run all tests from the project root folder by entering ` python -m unittest ` in your command line.
