#===============================================================
#; CREATION HISTORY:
#;       Written by:     Hui Xu/CI-SESS/UMD, Nov.31 2019
#;                       huixu@umd.edu
#; Purpose: SNPP/NOAA-20 CrIS cloud detection using DNN model
#;==============================================================
import os, sys, glob
import itertools
import sys
sys.path.append('/usr/bin/python/h5py')
import h5py
import numpy as np 



#read spectral cris files

cris_h5 = h5py.File(CrIS_SDR_FILENAME, 'r')
tmp_data = np.array(cris_h5['All_Data/CrIS-FS-SDR_All/ES_RealLW'][:])
orgdiminfo = tmp_data.shape
cris_rad_LW = hamming_apodization(tmp_data,twoD_reshape=1)
tmp_data = np.array(cris_h5['All_Data/CrIS-FS-SDR_All/ES_RealMW'][:])  
cris_rad_MW = hamming_apodization(tmp_data,twoD_reshape=1)
tmp_data = np.array(cris_h5['All_Data/CrIS-FS-SDR_All/ES_RealSW'][:])  
cris_rad_SW = hamming_apodization(tmp_data,twoD_reshape=1)
tensor = np.concatenate((cris_rad_LW, cris_rad_MW, cris_rad_SW), axis=1)


print(cris_rad_SW)




# RELU activation function
def relu(x):
    return np.maximum(0,x)
# SOFTMAX activation function
def softmax(x, axis=-1):
    x -= np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)
# Tensor forward propagation
def forward_propagation(tensor, weight, bias, activation='relu'):
    this_tensor = np.dot(tensor, weight) + bias
    if activation == 'relu':
        activation_func = relu
    elif activation == 'softmax':
        activation_func = softmax
    elif activation == 'pc':
        return this_tensor
    else:
        raise Exception('Non-supported function')       
    return activation_func(this_tensor)
# CrIS apodization in the spectral domain
def hamming_apodization(radiance, a=0.54, twoD_reshape=0):
    w0=(1.0-a)*0.5 # 0.23
    w1=a           # 0.54
    w2=(1.0-a)*0.5 # 0.23
    
    radiance = np.asarray(radiance, dtype=np.float64)
    orgdims  = radiance.shape
    radiance = radiance.reshape(-1, orgdims[-1]) 
    radianceApod = np.zeros_like(radiance)

    radianceApod[:, 0] = w1*radiance[:, 0] + w0*radiance[:, 1]
    radianceApod[:, 1:orgdims[-1]-1] = w0*radiance[:, 0:orgdims[-1]-2] + w1*radiance[:, 1:orgdims[-1]-1] + w2*radiance[:, 2:orgdims[-1]]
    radianceApod[:, orgdims[-1]-1] = w1*radiance[:, orgdims[-1]-1] + w0*radiance[:, orgdims[-1]-2]  
    radianceApod = radianceApod[:, 2:orgdims[-1]-2]
    # newdims = radianceApod.shape
    if twoD_reshape == 0:
        radianceApod = radianceApod.reshape(orgdims[:-1]+(-1, ))
    return radianceApod
# CrIS scene detection
def do_cris_scene_detection(CrIS_SDR_FILENAME, OUTPUT_FILENAME):
#{
    CLASS_TYPE = 'cloud_mask'
    """
    Method : do the cloud detection
    """
    DNN_COEF_FILENAME = './cris.fs.cloud.det.coefs.h5'
    # 2. READ DATA:
    # a) Read model coefficients  
    weight_list = []
    bias_list = []   
    # open h5 
    coef_h5 = h5py.File(DNN_COEF_FILENAME, 'r') 
    layer_names   = np.char.decode(np.array(coef_h5['layer_names'][:]), encoding='utf-8')
    layer_actfuns  = np.char.decode(np.array(coef_h5['layer_activations'][:]), encoding='utf-8')
    n_lay = layer_names.size # number of layers
    for k in np.arange(n_lay):
    #{
        info_tmp = layer_names[k].split("_")
        ds_name = 'model_weights/dense_' + info_tmp[1] + '/dense_' + info_tmp[1] + '_' + info_tmp[2] + '/kernel:0'  
        tmp_data = np.array(coef_h5[ds_name][:])
        weight_list.append(tmp_data)
        ds_name = 'model_weights/dense_' + info_tmp[1] + '/dense_' + info_tmp[1] + '_' + info_tmp[2] + '/bias:0'  
        tmp_data = np.array(coef_h5[ds_name][:]) 
        bias_list.append(tmp_data)
    #} 
    # close h5
    coef_h5.close()
    
    # b) Read CrIS SDR radiances
    # open h5 
    cris_h5 = h5py.File(CrIS_SDR_FILENAME, 'r')
    tmp_data = np.array(cris_h5['All_Data/CrIS-FS-SDR_All/ES_RealLW'][:])
    orgdiminfo = tmp_data.shape
    cris_rad_LW = hamming_apodization(tmp_data,twoD_reshape=1)
    tmp_data = np.array(cris_h5['All_Data/CrIS-FS-SDR_All/ES_RealMW'][:])  
    cris_rad_MW = hamming_apodization(tmp_data,twoD_reshape=1)
    tmp_data = np.array(cris_h5['All_Data/CrIS-FS-SDR_All/ES_RealSW'][:])  
    cris_rad_SW = hamming_apodization(tmp_data,twoD_reshape=1)
    tensor = np.concatenate((cris_rad_LW, cris_rad_MW, cris_rad_SW), axis=1)
    # close h5
    cris_h5.close()

    # 3. PREDICTION:
    for k in np.arange(n_lay): # tensor flowing
        tensor = forward_propagation(tensor, weight_list[k], bias_list[k], activation=layer_actfuns[k])
    if tensor.shape[-1] > 1:   # classification
        tensor_classes = tensor.argmax(axis=-1)
    else:
        tensor_classes = (tensor > 0.5).astype('int8')
    # reshpe to CrIS original shape
    tensor_classes = tensor_classes.reshape(orgdiminfo[:-1])

    # 4. OUTPUT (create h5 file)
    if os.path.isfile(OUTPUT_FILENAME):
        os.remove(OUTPUT_FILENAME)
    fid = h5py.File(OUTPUT_FILENAME, 'w')
    fid.create_dataset(CLASS_TYPE, data=tensor_classes)
    fid.close()
    print('writing into ' + OUTPUT_FILENAME)
#}    

# # # # # # # # # # # # # # # # # # # # # # #
#-Main function for the scene detection-#
# # # # # # # # # # # # # # # # # # # # # # #
if __name__ == '__main__':
# Example:
    # CrIS SDR file
    CrIS_SDR_FILENAME = './SCRIF_npp_d20170805_t0757519_e0758217_b29906_c20170805090427634658_nobc_ops.h5'
    # OUT file 
    OUTPUT_FILENAME = './SCRIF_npp_d20170805_t0757519_e0758217_b29906_c20170805090427634658_nobc_ops.cloudmask.h5'
    do_cris_scene_detection(CrIS_SDR_FILENAME, OUTPUT_FILENAME)