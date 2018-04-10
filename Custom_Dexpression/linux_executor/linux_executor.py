
# coding: utf-8

# In[1]:
print("pyhton script started")
import sys

# sys.path.insert(0, '/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/python2.7/site-packages')
# sys.path.insert(1,'/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/cuda_9.0/var/cuda-repo-9-0-local/usr/local/cuda-9.0/lib64')
# sys.path.insert(2,'/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/cudnn7/cudnn-9.0-linux-x64-v7/cuda/lib64')


# sys.path.insert(0, '/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/python2.7/site-packages')
sys.path.insert(0,'/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/cuda_9.0/var/cuda-repo-9-0-local/usr/local/cuda-9.0/lib64')
sys.path.insert(1,'/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/cudnn7/cudnn-9.0-linux-x64-v7/cuda/lib64')

print(sys.path)


import numpy as np
import tflearn
import tflearn.activations as activations
# Data loading and preprocessing
from tflearn.activations import relu
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import avg_pool_2d, conv_2d, max_pool_2d
from tflearn.layers.core import dropout, flatten, fully_connected, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.normalization import batch_normalization


#chris library imports
# from matplotlib import pyplot as plt
import cv2
import tensorflow as tf

from test_recursive_image_load_V2 import load_CKP_data
from test_recursive_image_load_V2 import load_formated_data
from showNumpyInfo import showInfo
from my_Dexpression_function import train_dexpression_model

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ID',type=int,
                   help='sum the integers (default: find the max)')

args = parser.parse_args()
# print(args.ID)



# In[ ]:



# global Paths to define for each specific computer
#tf_checkpoints = where the checkpoints of tensorflow training algorithms are stored to be recovered if necessary
tf_checkpoints = "./tf_checkpoints"

# cascPath = the path to the cascade file for the facerecognition (relative paths didn't work on my windows edition)
cascPath = "haarcascade.xml"

# Give a run ID here. Change it to flags (arguments) in version 2.
if args.ID == None:
    ID = '4_1'
else:
    ID = repr(args.ID)
    print(args.ID)

RUNID = 'DeXpression_run_' + ID

# Give a dropout if required (change to True and define the dropout percentage).
dropout_keep_prob=0.5

# Load data from: https://drive.google.com/drive/folders/1YWT8DJivNOZzQRPCiHDPY0LL_dymdQIS?usp=sharing
#if not in working directory look in alternative directory
try:
    X_data = np.load('CKP_X.npy')
    Y_data = np.load('CKP_Y.npy')
    X_subID = (np.load('CKP_subjectIds.npy')).astype('uint8')
except:
    X_data = np.load('../data/CKP_X.npy')
    Y_data = np.load('../data/CKP_Y.npy')
    X_subID = (np.load('../data/CKP_subjectIds.npy')).astype('uint8')

data = [X_data, Y_data, X_subID]  

print("CHECK data type and shape")

print("Type of data var ", type(data))
print("Type of data[0] ",  type(data[0]))

showInfo(X_data,"X_data")
showInfo(Y_data,"Y_data")
showInfo(X_subID ,"X_subID")
print("-----DONE DEBUG---------")

#load the subject distribution over the different datasets
subID = (np.load('data_division/train_subject_ID.npy')).astype('uint8')
subID_val = (np.load('data_division/validation_subject_ID.npy')).astype('uint8')
subID_test = (np.load('data_division/test_subject_ID.npy')).astype('uint8')
subIDs = [subID, subID_val, subID_test]

print("CHECK subID type and shape")

print("Type of data var ", type(data))
print("Type of data[0] ",  type(data[0]))
print("Type of subIDs var ", type(subIDs))
print("Type of subIDs[0] ",  type(subIDs[0]))
showInfo(subID,"subID")
showInfo(subID_val,"subID_val")
showInfo(subID_test ,"subID_test")
print("-----DONE DEBUG---------")


train_dexpression_model(RUNID,data,subIDs,tf_checkpoints,cascPath,dropout_keep_prob=0.5)
