import unittest 

from helpers import *

class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.path = "test/dummy_data.csv"

    def test_load_data(self):
        data = load_csv(self.path)
        self.assertEqual(data[0]['Id'], data[0][0]) #test dtype naming 
        self.assertEqual(data[0]["Prediction"],1) # test conversion of Prediction strings
        self.assertEqual(data.shape, (9,)) 
        print(data)


        


