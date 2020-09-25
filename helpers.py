import numpy as np

def load_csv(dataset_path, delimiter = ",", usecols = None):
    '''
    creates a structured np.array from the dataset where each datapoint will be a single tuple in the array, consisting of multiple named datatypes
    '''
    dtype = None # figure out the types by first trying Bool, the Int, then Float
    names = True # use first row as column names
    if usecols is not None:
        data = np.genfromtxt(dataset_path, dtype=dtype, delimiter=delimiter, names=names, usecols= usecols)
    else: 
        data = np.genfromtxt(dataset_path, dtype=dtype, delimiter=delimiter, names=names)

    return data
    