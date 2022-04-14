# Cloud-fraction-retrieval: Retrieve cloud fraction within CrIS FOVs
This project contains two programs including one model training program: cris_cloud_fraction_dnn_training.py, and one prediction program: cris_cloud_fraction_dnn_prediction.py. The detailed description of the two programs are as follows:
# 1. cris_cloud_fraction_dnn_training.py:
## The following codesimport common libs
import os, sys
import numpy as np
import h5py
from scipy import io as scipyIO
from netCDF4 import Dataset
from sklearn.metrics import mean_squared_error
import scipy.stats as st
## Import keras libs
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.models import model_from_json, load_model
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint
## Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
## Import sklearn libs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from shutil import copyfile
import matplotlib.pyplot as plt
## Define activation function
def custom_activation(x):
    return tf.minimum(tf.maximum(x, 0.0), 1.0)
class ReLU01(Activation):
    def __init__(self, activation, **kwargs):
        super(ReLU01, self).__init__(activation, **kwargs)
        self.__name__ = 'relu01'
def relu01(x):
    return tf.minimum(tf.maximum(x, 0.0), 1.0)

# 2. main functions
if __name__ == '__main__':
    
    get_custom_objects().update({'relu01': ReLU01(relu01)})  
 ## Define input and output path
    data_TRAINING_file = './cloudfrac_200PC_Training_ge4.sav' 
    model_name  = 'cloudfrac'
    output_dir = './cloudfrac/64_128_32'
 ## Define coefficinets such as total epoch number, learning rate, etc
    t_npc = 1   
    t_nep = 100
    save_every_coef_flag   = 1
    set_custom_lr = 1
    cus_lr = 0.001

    for ipc in range(t_npc):
        npc = 77
        model_dir   = output_dir + '/' + str(npc) + '_1th'    # model dir 
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)   
        stat_file = os.path.join(model_dir, 'training_stat.txt')
        timer_file = os.path.join(model_dir, 'timer.txt')
                            
  ## Data preparation and loading

        data = scipyIO.readsav(data_TRAINING_file)
        x_train = np.array(data['predictors'][:,0:npc]).astype(np.float)
        predictand_name = model_name
        y_train = np.array(data[predictand_name]).astype(np.float)
   

  ### code for training, evaluation and prediction are: Tranning=0, evaluating=1, predicting=2
    
        keras_all_in_one_file = os.path.join(model_dir, model_name+ '_pc_' + str(npc) +'.h5') 
        seed = 6
        np.random.seed(seed)
  ## Create training and testing data
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=seed)
        for i_epoch in range(t_nep):
            if os.path.exists(timer_file):
                f = open(timer_file, "r")
                tot_epochs = int(f.readline())
                f.close()
                tot_epochs +=1
            else:
                tot_epochs = 1
            print('Epoch %i/%i/%i'% (i_epoch+1, t_nep, tot_epochs))
            print('Build model ...')      
   ## build model file from already trained file
            if os.path.exists(keras_all_in_one_file):
                print('Updating coef file: ' )
                print(' ' + keras_all_in_one_file)
                def load_kerasRregressor_model():
                    model  = load_model(keras_all_in_one_file)
                    if set_custom_lr:
                        K.set_value(model.optimizer.learning_rate, cus_lr)
                    return model
                estimator = KerasRegressor(build_fn=load_kerasRregressor_model)
            else:
                print('Creating new coef file: ' )
                print(' ' + keras_all_in_one_file)
                loss_str = 'mean_squared_error' 
   ## Define optimizer 
                optimizer_str = 'adam'  
   ## define kerasClassifier model
                def generate_kerasRregressor_model():
   ## create model
                    model = Sequential()
                    model.add(Dense(units=64, input_dim=x_train.shape[1], init='normal', activation='relu'))  # input layer
                    model.add(Dropout(0.05))
                    model.add(Dense(units=128, init='normal', activation='relu'))
                    model.add(Dropout(0.05))
                    model.add(Dense(units=32, init='normal', activation='relu'))
                    model.add(Dropout(0.05))
                    model.add(Dense(units=1, init='normal', activation='relu'))
   ## Compile model
                    if set_custom_lr:
                        adam_opt = Adam(learning_rate=cus_lr)
                        model.compile(loss=loss_str, optimizer=adam_opt, metrics=['accuracy'])
                    else:
                        model.compile(loss=loss_str, optimizer=optimizer_str, metrics=['accuracy'])
                    return model
                estimator = KerasRegressor(build_fn=generate_kerasRregressor_model)                       
   ## Conduct training
            history = estimator.fit(x_train, y_train, validation_data=(x_test,y_test), verbose=True)
            acc  = np.array(history.history['accuracy'])[0]
            loss = np.array(history.history['loss'])[0]
            val_acc  = np.array(history.history['val_accuracy'])[0]
            val_loss = np.array(history.history['val_loss'])[0]
            score_str = "% s" % tot_epochs + ' ' + "%.6f" % (acc) + ' ' + "%.6f" % (loss) + ' ' + "%.6f" % (val_acc) + ' ' +  "%.6f" % (val_loss)          
            print('Training epoch finished:' )          
            print('Updating  the training coef file ...')
            estimator.model.save(keras_all_in_one_file)
            if save_every_coef_flag == 1:
                print('Save the coef and val files to:')
   ## Save the coefficients of the  model
                dirname = os.path.dirname(keras_all_in_one_file)
                filename= os.path.basename(keras_all_in_one_file)
                coef_dir = os.path.join(dirname, model_name+'_everycoefs')
                if not os.path.exists(coef_dir):
                    os.mkdir(coef_dir)      
                print(' ' + coef_dir)    
                filename= os.path.splitext(filename)[0]
                coef_file = os.path.join(coef_dir, filename+'.'+score_str+'.h5')
                copyfile(keras_all_in_one_file, coef_file)
   ## save the training accuracy results
            print('updating the training result file ...')
            print(' ' + stat_file)
            if os.path.exists(stat_file):
                f=open(stat_file, "a+")
            else:
                f= open(stat_file,"w+")
            f.write(score_str+ os.linesep)
            f.close()

            print(' ' + timer_file)
            if os.path.exists(timer_file):
                os.remove(timer_file)
            f= open(timer_file,"w+")
            f.write(str(tot_epochs))
            f.close()
