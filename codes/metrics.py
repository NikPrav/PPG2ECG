 from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error

import dataset_preprocessing as dp
import cv2
import module 
import preprocessing
import sklearn.preprocessing as skp
import tflib

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')
tf.autograph.set_verbosity(0)

@tf.function
def sample_P2E(P, model):
    fake_ecg = model(P, training=False)
    return fake_ecg

def read_file_and_estimate(path,model):
    ppg, ecg = dp.read_from_csv(path,'BIDMC')
    input_sampling_freq = 125
    ecg_sampling_freq = 128
    ppg_sampling_freq = 128
    window_size = 4
    ecg_segment_size = ecg_sampling_freq*window_size
    input_seg_size = input_sampling_freq*window_size
    loss_mse = 0
    loss_prd = 0
    k = 0
    for i in range(input_seg_size + 1,ppg.size,input_seg_size):
        x_ppg = ppg[i-input_seg_size:i]
        x_ecg = predict(x_ppg,model)
        ecg_resampled = cv2.resize(ecg[i-500:i], (1,ecg_segment_size), interpolation = cv2.INTER_LINEAR)
        ecg_resampled = ecg_resampled.reshape(1, -1)
        
        loss_mse += mean_squared_error(ecg_resampled, x_ecg)
        loss_prd += mean_absolute_percentage_error(ecg_resampled, x_ecg)
        k += 1
    
    return loss_mse/k, loss_prd/k
    
        

    
def predict(x_ppg,model):
    ppg_sampling_freq = 128
    window_size = 4
    ppg_segment_size = ppg_sampling_freq*window_size
    # resample to 128 Hz using: 
    x_ppg = cv2.resize(x_ppg, (1,ppg_segment_size), interpolation = cv2.INTER_LINEAR)
    # print(x_ppg.shape)
    x_ppg = x_ppg.reshape(1, -1)
    # filter the data using: 
    x_ppg = preprocessing.filter_ppg(x_ppg, 128)

    # make an array to N x 512 [this is the input shape of x_ppg], where Nx512=len(x_ppg)
    # normalize the data b/w -1 to 1: 
    x_ppg = skp.minmax_scale(x_ppg, (-1, 1), axis=1)
    # print(x_ppg)
    #######
    #
    x_ecg = sample_P2E(x_ppg, model)
    x_ecg = x_ecg.numpy()
    x_ecg = preprocessing.filter_ecg(x_ecg, 128)
    
    return x_ecg