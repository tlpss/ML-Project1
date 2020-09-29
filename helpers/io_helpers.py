import numpy as np

def load_csv(dataset_path, delimiter = ",", usecols = None, converters=None):
    '''
    creates a structured np.array from the dataset where each datapoint will be a single tuple in the array, consisting of multiple named datatypes
    '''
    dtype = None # figure out the types by first trying Bool, the Int, then Float
    names = True # use first row as column names
    
    if usecols is not None:
        data = np.genfromtxt(dataset_path, dtype=dtype, delimiter=delimiter, names=names, usecols= usecols,converters=converters )
    else: 
        data = np.genfromtxt(dataset_path, dtype=dtype, delimiter=delimiter, names=names,converters=converters)

    return data
    
def split_dataset(dataset, test_ratio = 0.1):
    '''
    expects as input a structured npdarray, with the first column being unique (an identifier)
    splits the dataset in train and testset according to the test_ratio 

    '''
    np.random.seed(2020) # fix seed for reproducability 
    shuffled_indices = np.random.permutation(len(dataset))
    test_set_size = int(len(dataset)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices =shuffled_indices[test_set_size:]


    return dataset[train_indices], dataset[test_indices]


def write_csv(data, filename):
    # create format array
    fmt = []
    for name in data.dtype.names:
        t = str(data.dtype[name])[0]
        if t is "i":
            fmt.append("%d")
        if t is "f":
            fmt.append("%3.3f")

     # create header
    header = ','.join(data.dtype.names)

    np.savetxt(filename, data, delimiter=',', header=header, fmt=fmt, comments="")


if __name__ == "__main__":

    convertfunc = lambda x: 0 if b'b' in x else 1 # convertfucntion for Prediction column to 0 if bg, and 1 if signal
    converters={"Prediction": convertfunc}
    dataset = load_csv("dataset/train.csv", converters=converters)
    train,test = split_dataset(dataset)
    write_csv(train, "dataset/trainset.csv")
    write_csv(test,"dataset/testset.csv")