import numpy as np

from helpers import *
from implementations import *
from preprocessing import *

###CONSTANTS####
np.random.seed(2020)
DEGREES = 9

###PREPROCESSING###
p_train = AddFeaturesPolyPreprocessing(load_csv('dataset/trainset.csv'))
p_train.set_degrees(DEGREES,1)
y_train , x_train= p_train.preprocess()
print(x_train.shape)

###TRAIN###
w_opt,loss = ridge_regression(y_train,x_train,0.00001)
print(loss)
###LOAD_PREDICTION_DATA###

p = AddFeaturesPolyPreprocessing(load_csv('dataset/test.csv'))
p.set_degrees(DEGREES,1)
ids, tx = p.preprocess(labeled=False)
###PREDICT###
pred = tx.dot(w_opt)
y = (pred > 0.5)*1
###CREATE SUBMISSION###
create_csv_submission(ids, y, f"submission.csv")