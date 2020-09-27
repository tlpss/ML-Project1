import unittest 

from helpers import load_csv, split_dataset, write_csv

class TestHelpers(unittest.TestCase):
    def setUp(self):
        self.path = "test/dummy_data.csv"

    def test_load_data(self):
        convertfunc = lambda x: 0 if b'b' in x else 1 # convertfucntion for Prediction column to 0 if bg, and 1 if signal
        converters={"Prediction": convertfunc}
        data = load_csv(self.path, converters=converters)

        self.assertEqual(data[0]['Id'], data[0][0]) #test dtype naming 
        self.assertEqual(data[0]["Prediction"],1) # test conversion of Prediction strings
        self.assertEqual(data.shape, (10,)) 
        #print(data)
        #print(data.dtype)


    def test_split_data(self):
        data = load_csv(self.path)

        trainset,testset = split_dataset(data, test_ratio=0.4)
        self.assertEqual(trainset.shape, (6,))
        self.assertEqual(testset.shape, (4,))

        #print(trainset)
        #print(testset)

    def test_write_csv(self):
        convertfunc = lambda x: 0 if b'b' in x else 1 # convertfucntion for Prediction column to 0 if bg, and 1 if signal
        converters={"Prediction": convertfunc}
        data = load_csv(self.path, converters=converters)
        write_csv(data,"test/test_write.csv")


