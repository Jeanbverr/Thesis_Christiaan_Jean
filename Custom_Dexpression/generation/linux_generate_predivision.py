
# coding: utf-8

# In[1]:

import sys

print("pyhton script started")

#sys.path.insert(0, '/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/python2.7/site-packages')
#sys.path.insert(1,'/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/cuda_9.0/var/cuda-repo-9-0-local/usr/local/cuda-9.0/lib64')
#sys.path.insert(2,'/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/cudnn7/cudnn-9.0-linux-x64-v7/cuda/lib64')

#sys.path.insert(0, '/esat/tiger/joramas/mscStudentsData/emotionModeling/libs/python2.7/site-packages')
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
from generate_predivided_data import  generate_predivided_data

generate_predivided_data()