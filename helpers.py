import numpy as np

def load_csv(dataset_path, delimiter = ",", usecols = None):
    '''
    creates a structured np.array from the dataset where each datapoint will be a single tuple in the array, consisting of multiple named datatypes
    '''
    dtype = None # figure out the types by first trying Bool, the Int, then Float
    names = True # use first row as column names
    convertfunc = lambda x: 0 if b'b' in x else 1 # convertfucntion for Prediction column to 0 if bg, and 1 if signal
    if usecols is not None:
        data = np.genfromtxt(dataset_path, dtype=dtype, delimiter=delimiter, names=names, usecols= usecols,converters={"Prediction": convertfunc} )
    else: 
        data = np.genfromtxt(dataset_path, dtype=dtype, delimiter=delimiter, names=names,converters={"Prediction": convertfunc})

    return data
    