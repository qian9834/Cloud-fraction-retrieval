# 0. Introduction
This project contains two programs including one model training program: cris_cloud_fraction_dnn_training.py, and one prediction program: cris_cloud_fraction_dnn_prediction.py. The detailed description of the two programs are as follows:
# 1. cris_cloud_fraction_dnn_training.py:
This program build DNN model using CRIS spectrum and VIIRS cloud data to retrieve the cloud fraction in each CrIS field of view
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
 ## Define input data and output path. The test data is stored in cloudfrac_200PC_Training_ge4.sav, there are two variables in this .sav file:1. "predcitors" which contains 200 pricipal components serving as the model input predictors; 2. "cloudfrac" which is VIIRS cloud fractions serving as model learning target.
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
                            
  ## Data preparation and loading. Load the first 77 PCs from "predictors" variable, and VIIRS cloud fraction from the "cloudfrac" variable.

        data = scipyIO.readsav(data_TRAINING_file)
        x_train = np.array(data['predictors'][:,0:npc]).astype(np.float)
        predictand_name = model_name
        y_train = np.array(data[predictand_name]).astype(np.float)

  ### Code for training, evaluation and prediction are: Tranning=0, evaluating=1, predicting=2
    
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
   ## Save the coefficients of the  model. There is a previously trained model coefficient saved in the "cloudfrac.h5" in this project, which is produced in this step.
                dirname = os.path.dirname(keras_all_in_one_file)
                filename= os.path.basename(keras_all_in_one_file)
                coef_dir = os.path.join(dirname, model_name+'_everycoefs')
                if not os.path.exists(coef_dir):
                    os.mkdir(coef_dir)      
                print(' ' + coef_dir)    
                filename= os.path.splitext(filename)[0]
                coef_file = os.path.join(coef_dir, filename+'.'+score_str+'.h5')
                copyfile(keras_all_in_one_file, coef_file)
   ## Save the training accuracy results
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
         
# 2. cris_cloud_fraction_dnn_prediction.py
This program produce cloud fraction retrieval results using previously built DNN model.
## Import common libs
    import os, sys
    import numpy as np
    import h5py
    from scipy import io as scipyIO
    from netCDF4 import Dataset
## Import keras libs
    from keras.models import Sequential
    from keras.layers import Dense
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

## Define input data and model path and output path. The pre-trained model is "cloudfrac.h5" in this project folder.
    data_file    = './test_data/cris_viirs_cloudfrac_input_20200601.sav'  # input file
    model_dir  = './cloudfrac'         # model dir
    model_name  = 'cloudfrac'       # model name
    npc  = 77             # number of predictors
    output_file = './test_data/cris_viirs_cloudfrac_output_20200601.h5'   # output file

## Prepare and load dataset from "predictors" variable of the sample data.
    data = scipyIO.readsav(data_file)
    predictors = np.array(data['predictors'][:,0:npc]).astype(np.float)
    predictand_name = model_name
## Load DNN model from "cloudfrac.h5"
    keras_all_in_one_file = os.path.join(model_dir, model_name+'.h5') 
    estimator = load_model(keras_all_in_one_file)
    predictions = np.squeeze(np.array(estimator.predict(predictors)))
## Solving values outside[0,1], set > 1.0 as 1.0 and set <0.0 as 0.0 
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
  ## output the result (create h5 file)
    if os.path.isfile(output_file):
        fid = h5py.File(output_file, 'a')
    else:
        fid = h5py.File(output_file, 'w')
 ## create dimension variable
    fid.create_dataset(model_name + '_p', data=predictions)
 ## close the nc file
    fid.close()
 # Reference
1.Liu, Q., Li, Y., Yu, M., Chiu, L.S., Hao, X., Duffy, D.Q. and Yang, C., 2019. Daytime rainy cloud detection and convective precipitation delineation based on a deep neural Network method using GOES-16 ABI images. Remote Sensing, 11(21), 2555.
2.Liu, Q., Xu, H., Sha, D., Lee, T., Duffy, D.Q., Walter, J. and Yang, C., 2020. Hyperspectral Infrared Sounder Cloud Detection Using Deep Neural Network Model. IEEE Geoscience and Remote Sensing Letters. 19 (2022), 5500705
3.Liu, Q., Chiu, L.S., Hao, X. and Yang, C., 2021. Spatiotemporal Trends and Variations of the Rainfall Amount, Intensity, and Frequency in TRMM Multi-satellite Precipitation Analysis (TMPA) Data. Remote Sensing, 13(22), 4629.
![image](https://user-images.githubusercontent.com/39736687/184898244-a6f73c4b-8f78-4199-857d-57e796116b78.png)
4. Xu, H., Chen, Y. and Wang, L., 2018. Cross-track infrared sounder spectral gap filling toward improving intercalibration uncertainties. IEEE Transactions on Geoscience and Remote Sensing, 57(1), 509-519.
5. Yang, C., Yu, M., Li, Y., Hu, F., Jiang, Y., Liu, Q., Sha, D., Xu, M. and Gu, J., 2019. Big Earth data analytics: A survey. Big Earth Data, 3(2), 83-107.
6. Zou, C.Z., Zhou, L., Lin, L., Sun, N., Chen, Y., Flynn, L.E., Zhang, B., Cao, C., Iturbide-Sanchez, F., Beck, T. and Yan, B., 2020. The reprocessed Suomi NPP satellite observations. Remote Sensing, 12(18), 2891. 

![image](https://user-images.githubusercontent.com/39736687/184898391-3e713ecf-9bac-4204-94ff-618014675178.png)


