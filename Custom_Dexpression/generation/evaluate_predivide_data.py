import numpy as np
import os
import sys

# adds the lower lying directory to the import path to import the other modules
Lpath = os.path.abspath('..')
print("found path with os.path.abspath('..'): ", Lpath)
sys.path.insert(0, Lpath)

#chris library imports
# from matplotlib import pyplot as plt
import cv2
# import tensorflow as tf
from sklearn.model_selection import train_test_split


from test_recursive_image_load_V2 import load_CKP_data
from test_recursive_image_load_V2 import load_formated_data
from test_recursive_image_load_V2 import split_dataset
from test_recursive_image_load_V2 import divide_subjects
from test_recursive_image_load_V2 import divide_data_to_subject
from test_recursive_image_load_V2 import load_npy_files

import logging
from memory_profiler import memory_usage

load_dir = 'CKP'

[X_data,Y_data,X_subID] = load_npy_files(5,load_dir)
data = [X_data,Y_data,X_subID]

indices = np.argsort(X_subID,kind='mergesort',axis = 0)

X_data = X_data[indices]
Y_data = Y_data[indices]
X_subID = X_subID[indices]


X = np.load('./predivided_data/'+load_dir+'/X_train.npy')
Y = np.load('./predivided_data/'+load_dir+'/Y_train.npy')

X_val = np.load('./predivided_data/'+load_dir+'/X_val.npy')
Y_val = np.load('./predivided_data/'+load_dir+'/Y_val.npy')

X_test = np.load('./predivided_data/'+load_dir+'/X_test.npy')
Y_test = np.load('./predivided_data/'+load_dir+'/Y_test.npy')

print( "tot length ", len(X) + len(X_val) + len(X_test))
print("origin length", len(X_data))

i_data = 0
i = 0

for img in X:
	name1 = "X "+repr(i)
	cv2.namedWindow(name1)
	cv2.moveWindow(name1, 100,0)
	cv2.imshow(name1,img)
	name = "X_data "+repr(i_data)
	cv2.namedWindow(name)
	cv2.moveWindow(name, 400,0)
	cv2.imshow(name,X_data[i_data].astype('uint8'))

	i = i +1
	i_data = i_data +1
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

i = 0

for img in X_val:
	name1 = "X_val "+repr(i)
	cv2.namedWindow(name1)
	cv2.moveWindow(name1, 100,0)
	cv2.imshow(name1,img)
	name = "X_data "+repr(i_data)
	cv2.namedWindow(name)
	cv2.moveWindow(name, 400,0)
	cv2.imshow(name,X_data[i_data].astype('uint8'))

	i = i +1
	i_data = i_data +1
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()

i = 0

for img in X_test:
	name1 = "X_test "+repr(i)
	cv2.namedWindow(name1)
	cv2.moveWindow(name1, 100,0)
	cv2.imshow(name1,img)
	name = "X_data "+repr(i_data)
	cv2.namedWindow(name)
	cv2.moveWindow(name, 400,0)
	cv2.imshow(name,X_data[i_data].astype('uint8'))

	i = i +1
	if(i_data < len(X_data)-1):
		i_data = i_data +1
	
	cv2.waitKey(0)
	cv2.destroyAllWindows()