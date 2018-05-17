
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
from sklearn.model_selection import train_test_split


from test_recursive_image_load_V2 import load_CKP_data
from test_recursive_image_load_V2 import load_formated_data
from test_recursive_image_load_V2 import split_dataset
from test_recursive_image_load_V2 import divide_subjects
from test_recursive_image_load_V2 import divide_data_to_subject
from test_recursive_image_load_V2 import load_npy_files

from showNumpyInfo import showInfo

from Dexpression_network import create_Dexpression_network
from Dexpression_network import create_Dexpression_GAP_network
from Dexpression_network import create_original_Dexpression_network

def train_dexpression_model(RUNID,data,subIDs,tf_checkpoints,cascPath,dropout_keep_prob=0.5):
    
    print("CHECK data types and shapes")
    print("length of data list: ",len(data))
    print("length of subId list: ", len(subIDs))
    print("length of data list: ",len(data))
    
    print("Type of data var ", type(data))
    print("Type of data[0] ",  type(data[0]))
    print("Type of subIDs var ", type(subIDs))
    print("Type of subIDs[0] ",  type(subIDs[0]))
    
    showInfo(data[0],"X_data")
    showInfo(data[1],"Y_data")
    showInfo(data[2],"X_subID")
    showInfo(subIDs[0],"subID")
    showInfo(subIDs[0],"subID_val")
    showInfo(subIDs[0] ,"subID_test")
    print("-----DONE DEBUG---------")
    
    # divides the data in training, validation and test sets according to lists of the subjectIDs already divided over the 3
    # IN:
    # data = contain 3 1D-arrays: x,y and subject data [X_data, Y_data,X_subID]
    # subIDs = contain 3 1D-arrays with the subject numbers for each set: train,val,test [subID subID_val subID_test]
    # OUT:
    # list of 6 arrays [X,Y,X_val,Y_val,X_test,Y_test]
    divided_data = divide_data_to_subject(data,subIDs)

    X = (divided_data[0].reshape(-1,224,224,1)).astype('uint8')
    Y = (divided_data[1].reshape(-1,7)).astype('uint8')

    # create the validation set X_val and Y-val (SubID_val is not given to the network)
    X_val = divided_data[2].reshape(-1,224,224,1).astype('uint8')
    Y_val = divided_data[3].reshape(-1,7).astype('uint8')

    # create the test set X_test and Y_test (SubID_test is not given to the network)
    X_test = divided_data[4].reshape(-1,224,224,1).astype('uint8')
    Y_test = divided_data[5].reshape(-1,7).astype('uint8')

#     network = create_Dexpression_network(dropout_keep_prob)

#     network with GAP layer
    network = create_Dexpression_network(dropout_keep_prob)
    


    #create a custom tensorflow session to manage the used resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)


    # Final definition of model checkpoints and other configurations
    #model = tflearn.DNN(network, checkpoint_path='/home/cc/DeXpression/DeXpression_checkpoints',
    model = tflearn.DNN(network, checkpoint_path=tf_checkpoints,
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir="./tflearn_logs/")


    # Fit the model, train for 20 epochs. (Change all parameters to flags (arguments) on version 2.)
    #model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True, show_metric=True, batch_size=50, snapshot_step=2000,snapshot_epoch=True, run_id=RUNID)
    model.fit(X, Y, n_epoch=20, validation_set=(X_val,Y_val), shuffle=True, show_metric=True, batch_size=50, snapshot_step=2000,snapshot_epoch=True, run_id=RUNID)

    # Save the model
    model.save(tf_checkpoints + '/' + RUNID + '.model')
    print("finished training and saving")



# Data is formated as an list of [X_train , Y_train, X_val, Y_val, X_test, Y_test]
def train_on_predivided_dexpression_model(RUNID,data,tf_checkpoints,cascPath,dropout_keep_prob=0.5):
    
    print("CHECK data types and shapes")
    print("length of data list: ",len(data))
    print("Type of data var ", type(data))
    print("Type of data[0] ",  type(data[0]))

    
    showInfo(data[0],"X_train")
    showInfo(data[1],"Y_train")
    showInfo(data[2],"X_val")
    showInfo(data[3],"Y_val")
    showInfo(data[4],"X_test")
    showInfo(data[5],"Y_test")

    print("-----DONE DEBUG---------")
    

    X = (data[0].reshape(-1,224,224,1)).astype('uint8')
    Y = (data[1].reshape(-1,7)).astype('uint8')

    # create the validation set X_val and Y-val (SubID_val is not given to the network)
    X_val = data[2].reshape(-1,224,224,1).astype('uint8')
    Y_val = data[3].reshape(-1,7).astype('uint8')

    # create the test set X_test and Y_test (SubID_test is not given to the network)
    X_test = data[4].reshape(-1,224,224,1).astype('uint8')
    Y_test = data[5].reshape(-1,7).astype('uint8')

#     network = create_Dexpression_network(dropout_keep_prob)

#     network with GAP layer
    network = create_original_Dexpression_network(dropout_keep_prob)
    


    #create a custom tensorflow session to manage the used resources
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config = config)


    # Final definition of model checkpoints and other configurations
    #model = tflearn.DNN(network, checkpoint_path='/home/cc/DeXpression/DeXpression_checkpoints',
    model = tflearn.DNN(network, checkpoint_path=tf_checkpoints,
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir="./tflearn_logs/")


    # Fit the model, train for 20 epochs. (Change all parameters to flags (arguments) on version 2.)
    #model.fit(X, Y, n_epoch=20, validation_set=0.1, shuffle=True, show_metric=True, batch_size=50, snapshot_step=2000,snapshot_epoch=True, run_id=RUNID)
    model.fit(X, Y, n_epoch=20, validation_set=(X_val,Y_val), shuffle=True, show_metric=True, batch_size=50, snapshot_step=2000,snapshot_epoch=True, run_id=RUNID)

    # Save the model
    model.save(tf_checkpoints + '/' + RUNID + '.model')
    print("finished training and saving")

