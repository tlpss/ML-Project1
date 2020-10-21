import unittest
import numpy as np

from implementations import * 

class TestImplementations(unittest.TestCase):
    def setUp(self):
        self.w  = np.array([1.3, 2.2, 1.8])
        self.N = 1000
        self.y, self.y_ridge, self.tX = TestImplementations.generate_dataset(self.w, self.N)

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
        sig = 1/(1 + np.exp(-y))
        y += np.random.normal(0, 0.01, len(y)) # add noise 
        
        f = lambda x : float(0) if x <0.5 else float(1)
        y_ridge = np.array([ f(x) for x in sig])
        #y_ridge+= np.random.normal(0, 0.01, len(y_ridge)) # add noise 

        return y, y_ridge, X

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
        
    def test_reg_logistic_regression_extremes(self):
        #print(self.y_ridge)
        w,loss = reg_logistic_regression(self.y_ridge, self.tX,0.2,np.array([0,0,0]), 100, 0.01)
        y_ridge = np.dot(self.tX,w)
        #print("Loss", loss)
        sig = 1/(1 + np.exp(-y_ridge))
        f = lambda x : float(0) if x <0.5 else float(1)
        y_ridge = np.array([ f(x) for x in sig])
        #print(y_ridge == self.y_ridge)
        diff = y_ridge == self.y_ridge
        mean = np.mean(diff*1)
        self.assertTrue(mean > 0.97) # accuracy should be very high

    def test_confusion_matrix(self):
        y = np.array([1,1,1,0,0,0,1])
        y_hat = np.array([1,1,1,1,0,0,1])

        c = confusion_matrix(y,y_hat)
        print(c)
        self.assertAlmostEqual(c[0,0],2/3)
        self.assertAlmostEqual(c[1,1],3/3)
        self.assertAlmostEqual(c[0,1],1/3)
        self.assertAlmostEqual(c[1,0],0/3)

 
        
        

