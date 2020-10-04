import unittest
import numpy as np

from implementations import * 

class TestImplementations(unittest.TestCase):
    def setUp(self):
        self.w  = np.array([1.3, 2.2, 1.8])
        self.N = 1000
        self.y, self.tX = TestImplementations.generate_dataset(self.w, self.N)

    @staticmethod
    def generate_dataset(w, N):
        """
        generate y, tX of random dataset with specified relation

        :param w: [description]
        :type w: [type]
        :param N: [description]
        :type N: [type]
        :return: [description]
        :rtype: [type]
        """
        X = np.random.normal(0,1,(N,len(w)-1)) #create datapoints 
        bias_vec = np.ones((X.shape[0],1))
        X = np.concatenate((bias_vec, X), axis = 1)
        y = np.dot(X,w)
        y += np.random.normal(0, 0.01, len(y)) # add noise 

        return y, X

    def test_SGD(self):
        w, loss = least_squares_SGD(self.y, self.tX, np.array([0,0,0]), 50, 0.1, 100)
        #print(w)
        diff = w - self.w
        #print(diff)
        self.assertTrue(np.dot(diff,diff.T) < 0.01) # should not be too far from "exact weight vector"

    def test_normal_least_squares(self):
        w, loss = normal_least_squares(self.y,self.tX)
        diff = w - self.w
        #print(w)
        self.assertTrue(np.dot(diff,diff.T) < 0.01) # should not be too far from "exact weight vector"

    def test_ridge_extremes(self):
        w,loss = ridge_regression(self.y, self.tX,100000000)
        #print(w)
        self.assertTrue(np.dot(w,w.T) < 0.001)
        w, loss = ridge_regression(self.y, self.tX, 0)
        diff = w - self.w
        #print(w)
        self.assertTrue(np.dot(diff,diff.T) < 0.01) # should not be too far from "exact weight vector"

