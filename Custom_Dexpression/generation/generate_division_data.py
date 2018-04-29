import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dir',type=str,
                   help='directory name to get the input data and the directory where to store the results in, example python generate_division_data.py CKP_all_neutral')

args = parser.parse_args()
# print(args.ID)
dire = args.dir


import numpy as np
import os
import sys

# adds the lower lying directory to the import path to import the other modules
Lpath = os.path.abspath('..')
print("found path with os.path.abspath('..'): ", Lpath)
sys.path.insert(0, Lpath)

from test_recursive_image_load_V2 import  split_subject_dataset
from test_recursive_image_load_V2 import  divide_subjects
from test_recursive_image_load_V2 import  load_npy_files

import logging
from memory_profiler import memory_usage

new = False

if not os.path.exists('../data_division/'+dire):
    os.makedirs('../data_division/'+dire)
    new = True
    
logfile = '../data_division/'+dire+'/division_info.txt'

# if __name__ == "__main__":
if not os.path.exists('../data_division/'+dire +'/' + logfile):
	os.remove(logfile)
logging.basicConfig(level=logging.DEBUG, filename = logfile, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info("start logging to " + logfile)

def logprint(stre):
	print(stre)
	logging.info(stre)

if(new == True):
	logprint("made new directory: " + '../data_division/'+ dire )
# Author: Christiaan Vanbergen
# This script generates a division of subjects of a dataset to be later used to divide the input and output data
# over train,validation and test data.
# To generate a custom division change the inputs of split_dataset and the select_* variables
# generate_data.py can generate the X.npy, Y.npy,subjectIDs.npy loaded here
# Or the files can be found on: https://drive.google.com/drive/folders/1YWT8DJivNOZzQRPCiHDPY0LL_dymdQIS?usp=sharing

[X_data,Y_data,X_subID] = load_npy_files(5,dire)


# seperate the dataset in 10 parts that do not share users: 
#   6 parts train set
#   3 part validation set
#   1 part test set
# allows future possibility for 10-fold crossover

X_subID_parts = split_subject_dataset(X_subID,10,logfile)

select =  [0,1,2,3,4,5] # 60%
select_val = [6,7,8] 	# 30%
select_test= [9]		# 10%

selection = [select, select_val, select_test]

lenSelect = len(select)
lenSelect_val = len(select_val)
lenSelect_test = len(select_test)
lenTot = lenSelect + lenSelect_val + lenSelect_test
lenParts = len(X_subID_parts)

if(lenTot != lenParts):
	logprint("Warning it is possible that not all parts of the dataset are used")

[subID, subID_val, subID_test] =  divide_subjects(X_subID_parts,selection, logfile)

logprint("there are " + repr(lenParts) + " parts")
logprint("the training   set contains parts " + repr(select) + " that is " + repr(lenSelect) + " of those parts which is " + repr((lenSelect/lenTot)*100) + "%")
logprint("the validation set contains parts " + repr(select_val) + " that is " + repr(lenSelect_val) + " of those parts which is " + repr((lenSelect_val/lenTot)*100) + "%")
logprint("the test       set contains parts " + repr(select_test) + " that is " + repr(lenSelect_test) + " of those parts which is " + repr((lenSelect_test/lenTot)*100) + "%")


np.save('../data_division/'+dire+'/train_subject_ID.npy',subID)
np.save('../data_division/'+dire+'/validation_subject_ID.npy',subID_val)
np.save('../data_division/'+dire+'/test_subject_ID.npy',subID_test)

logprint ("generated division files in ../data_division/"+dire)
