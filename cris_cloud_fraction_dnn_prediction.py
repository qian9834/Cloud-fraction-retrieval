# Author: Qian Liu - George Masion University
# Email: qliu6@gmu.edu

# import common libs
import os, sys
import numpy as np
import h5py
from scipy import io as scipyIO
from netCDF4 import Dataset
# import keras libs
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json, load_model
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint
# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
# import sklearn libs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from shutil import copyfile
import matplotlib.pyplot as plt

def custom_activation(x):
    return tf.minimum(tf.maximum(x, 0.0), 1.0)
class ReLU01(Activation):
    def __init__(self, activation, **kwargs):
        super(ReLU01, self).__init__(activation, **kwargs)
        self.__name__ = 'relu01'
def relu01(x):
    # Your activation function specialties here
    return tf.minimum(tf.maximum(x, 0.0), 1.0)

# # # #  # # # # # # # # # # # # # # # # # # # #  # # # # #
# # # #  ###### using Keras  ######  # # # # 
# # # #  # # # # # # # # # # # # # # # # # # # #  # # # # # 
# main functions
if __name__ == '__main__':
    get_custom_objects().update({'relu01': ReLU01(relu01)})
    # input from the command line
    data_file    = './test_data/cris_viirs_cloudfrac_input_20200601.sav'  # input file
    model_dir  = './cloudfrac'         # model dir
    model_name  = 'cloudfrac'       # model name
    npc  = 77             # number of predictors
    output_file = './test_data/cris_viirs_cloudfrac_output_20200601.h5'   # output file

# <I>. DATA PREPARATION 
#{
    # load dataset
    data = scipyIO.readsav(data_file)
    predictors = np.array(data['predictors'][:,0:npc]).astype(np.float)
    predictand_name = model_name
#}
    keras_all_in_one_file = os.path.join(model_dir, model_name+'.h5') 
    estimator = load_model(keras_all_in_one_file)
    predictions = np.squeeze(np.array(estimator.predict(predictors)))
    # solving values outside[0,1]
    # Correction : set > 1.0 as 1.0 and set <0.0 as 0.0 
    invalid = np.where(predictions >1.0)
    invalicCT = len(invalid[0])
    if invalicCT >0:
        print('Number of points gt 1.0:')
        print(invalicCT)
        predictions[invalid] = 1.0
    invalid = np.where(predictions <0.0)
    invalicCT = len(invalid[0])
    if invalicCT >0:
        print('Number of points lt 0.0:')
        print(invalicCT)
        predictions[invalid] = 0.0     
    # output the result (create h5 file)
    if os.path.isfile(output_file):
        fid = h5py.File(output_file, 'a')
    else:
        fid = h5py.File(output_file, 'w')
    # create dimension variable
    fid.create_dataset(model_name + '_p', data=predictions)
    # close the nc file
    fid.close()
#}
