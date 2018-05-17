
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys

# adds the lower lying directory to the import path to import the other modules
Lpath = os.path.abspath('..')
print("found path with os.path.abspath('..'): ", Lpath)
sys.path.insert(0, Lpath)

from test_recursive_image_load_V2 import load_npy_files
import logging
from memory_profiler import memory_usage

import argparse

parser = argparse.ArgumentParser(description='generate data infor frm .npy files in the given directory')
parser.add_argument('dir',type=str,
                   help='the data directory from which to generate the info')

args = parser.parse_args()



# In[2]:


load_dir = args.dir #the directory that is created in the predivided_data directory






# In[3]:


def logprint(*arg):
    string = "str"
    stre = ""
    for a in arg:
        if(type(a) == type(string)):
            stre = stre + " " +  a
        else:
            stre = stre + " " + repr(a)
    print(stre)
    logging.info(stre)

def showInfo(Var,VarName = "UnKnown"):
    logprint("----------------------")
    logprint("Name " + repr(VarName))
    logprint("type: " + repr(type(Var)))
    logprint("Dtype: " + repr(Var.dtype))
    logprint("shape: " + repr(Var.shape))
    
def extract_classes(DNN_output):
    classes = []  
    for lab in DNN_output:
        maxi = lab.max()
        for i in range(0,len(lab)):
            if(lab[i] == maxi):
                classes.append(i)
    return np.asarray(classes)


# In[4]:


def calculate_emotion_distru(Y_data):
    Y_classes = extract_classes(Y_data)
    [A,N,D,F,H,Sa,Su] = np.bincount(Y_classes)
    i = np.sum([A,N,D,F,H,Sa,Su])
    
    A = int(A)
    N = int(N)
    D = int(D)
    F = int(F)
    H = int(H)
    Sa = int(Sa)
    Su = int(Su)
    
    return i,A,N,D,F,H,Sa,Su

def number_of_subjects(X_subID):
    return ((np.unique(X_subID)).shape[0])
    
def log_data_info(load_dir,data)

    [X_data, Y_data, X_subID] = data

    # log the general size info of the arrays
    logprint("Dataset name = ", load_dir )
    logprint("General Size of arrays info")
    showInfo(X_data,'X_data')
    showInfo(Y_data,'Y_data')
    showInfo(X_subID,'X_subID')
    logprint("")


    # In[7]:


    # log the amount of memory used
    mem_usage = memory_usage(-1, interval=1, timeout=1)
    str = 'Maximum memory ' + repr( mem_usage)
    logprint(str)
    logprint("")


    # In[8]:


    sub = number_of_subjects(X_subID)
    i,A,N,D,F,H,Sa,Su = calculate_emotion_distru(Y_data)


    #  the distribution of the emotions in the dataset
    logprint("the distribution of the emotions in the dataset")
    logprint("--------- Overal stattistics ---------  ")
    logprint("amount of Subjects  : " + repr(sub))
    logprint("amount of Instances : " + repr(i))
    logprint("index = 0 = Anger      " + repr(A) + " instances: " + repr(np.round((float(A)/i)*100,decimals=2)))
    logprint("index = 1 = Neutral    " + repr(N) + " instances: " + repr(np.round((float(N)/i)*100,decimals=2)))
    logprint("index = 2 = Disgust    " + repr(D) + " instances: "+ repr(np.round((float(D)/i)*100,decimals=2)))
    logprint("index = 3 = Fear       " + repr(F) + " instances: "+ repr(np.round((float(F)/i)*100,decimals=2)))
    logprint("index = 4 = Happy      " + repr(H) + " instances: "+ repr(np.round((float(H)/i)*100,decimals=2)))
    logprint("index = 5 = Saddness   " + repr(Sa) +" instances: "+ repr(np.round((float(Sa)/i)*100,decimals=2)))
    logprint("index = 6 = Surprise   " + repr(Su) +" instances: "+ repr(np.round((float(Su)/i)*100,decimals=2)))

    logprint("SUCCESFULL Creation of data_info for " +  load_dir)



# In[5]:

def generate_data_info(load_dir):

    # configure the logging    
    logfile = '../../data/'+load_dir+'/data_info.txt'

    if os.path.exists(logfile):
        os.remove(logfile)

     # needs to be run to make sure the logfile is created in basicConfig
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(level=logging.DEBUG, filename = logfile, filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("start logging to " + logfile)



    #  load the dataset
    data = load_npy_files(5,load_dir)


    # log text
    log_data_info(load_dir,data)


    

if __name__ == '__main__':
    generate_data_info(load_dir)