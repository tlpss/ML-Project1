# ML-Project1
Project 1 of the ML course @ EPFL (CS-433)




## Project Structure
TODO

## Coding Style
Documentation according to the **docstring - reST** style 

see https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html for a small example


## Testing
Testing is done using [Unittest](https://docs.python.org/3/library/unittest.html), which is the built-int test framework for python. 

Tests are grouped in the `/test` folder. For tests to be auto-discoverable by the framework the files need to start with `test` and the tests need to be in a class that inherits from the `unittest.TestCase` baseclass. 

You can then run all tests from the project root folder by entering ` python -m unittest ` in your command line.
