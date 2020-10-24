import numpy as np

class Preprocessing:
    """
    Implements a preprocessing pipeline
    """
    def __init__(self, dataset):
        """
        :param dataset: the training set 
        :type dataset: structured numpy array, containg an "ID" and "Predictions" Column as first two columns
        all values are numerical and missing values are as specified 
        """
        self.MISSING_VALUE = -999.0

        self.original_dataset = dataset
        self.dataset = dataset
        self.y = None
        self.tX = None

    def preprocess(self, replace_missing_by_mean = True, outlier_removal = False, save_y_and_tX = False, labeled = True):
        """
        performs the steps in the pipepline

        
        :return: y, tX (extended!)
        :rtype: numpy array, 2D-numpy array 
        """
        self._split_predictions_and_make_unstructured(labeled=labeled)
        if replace_missing_by_mean:
            self._replace_missing_by_mean()
            
        self._feature_transformation()
        
        self._standardize_columns()

        self._feature_engineering()
        if outlier_removal:
            self._remove_outliers()
        self._add_bias_column_and_create_tX()

        if save_y_and_tX:
            np.save('dataset/preprocessed_tx.npy',self.tX)
            np.save('dataset/y.npy', y)
        return self.y, self.tX


    def _split_predictions_and_make_unstructured(self, labeled):
        if labeled:
            self.y  = self.dataset["Prediction"]
            self.dataset = np.array(self.dataset.tolist()) # make unstructured, not very efficient..
            self.dataset = self.dataset[:,2:] # filter out id and prediction
        else:
            print("prediction data")
            self.y = self.dataset["Id"] # store IDs
            self.dataset = np.array(self.dataset.tolist()) # make unstructured, not very efficient..
            self.dataset = self.dataset[:,2:] # remove IDs and '?' of predictions

    
    def _replace_missing_by_mean(self):
        # create masked ndarray to discard missing values
        # https://numpy.org/doc/stable/reference/maskedarray.generic.html#constructing-masked-arrays
        mask = self.dataset == self.MISSING_VALUE
        masked_data = np.ma.array(self.dataset)
        masked_data.mask = mask
        masked_data

        # calc means

        means = np.ma.mean(masked_data, axis = 0)
        self.feature_means = means

        # replace missing values
        indices = np.where(self.dataset == self.MISSING_VALUE)
        self.dataset[indices] = np.take(means, indices[1])

        assert np.sum(self.dataset == self.MISSING_VALUE) == 0
        assert np.sum((np.mean(self.dataset, axis =0)-means)**2) < 0.0001

    def _remove_outliers(self):
        raise NotImplementedError
    
    def _feature_transformation(self):
        ''' Method to be implemented by subclasses'''
        pass

    def _feature_engineering(self):
        ''' Method to be implemented by subclasses'''
        pass

    def _standardize_columns(self):
        self.dataset = (self.dataset- np.mean(self.dataset, axis = 0)) / np.std(self.dataset, axis = 0)

    def _add_bias_column_and_create_tX(self):
        bias_vec = np.ones((self.dataset.shape[0],1))
        self.tX = np.concatenate((bias_vec,self.dataset), axis = 1)

if __name__ == "__main__":
    """ example usage"""
    
    from helpers.io_helpers import load_csv
    #p = Preprocessing(load_csv("dataset/trainset.csv"))
    #y, tX = p.preprocess()
    p = Preprocessing(load_csv("dataset/unlabeled_test_0.csv", converters={}))
    ids, tX = p.preprocess(labeled=False)

    
