import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('dir',type=str,
                   help='directory name to get the input data and the directory where to store the results in, example python generate_division_data.py CKP')

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

# Author: Christiaan Vanbergen
# This script generates a division of subjects of a dataset to be later used to divide the input and output data
# over train,validation and test data.
# To generate a custom division change the inputs of split_dataset and the select_* variables
# generate_data.py can generate the X.npy, Y.npy,subjectIDs.npy loaded here
# Or the files can be found on: https://drive.google.com/drive/folders/1YWT8DJivNOZzQRPCiHDPY0LL_dymdQIS?usp=sharing

[X_data,Y_data,X_subID] = load_npy_files(5,dire)


# seperate the dataset in 10 parts that do not share users: 
#   8 parts train set
#   1 part validation set
#   1 part test set
# allows future possibility for 10-fold crossover

X_subID_parts = split_subject_dataset(X_subID,10)

select =  [0,2,5,6,8,9] # 60%
select_val = [4,7,3] 	# 30%
select_test= [1]		# 10%

selection = [select, select_val, select_test]

if((len(select)+len(select_val) + len(select_test))!= len(X_subID_parts)):
	print("Warning it is possible that not all parts of the dataset are used")

[subID, subID_val, subID_test] =  divide_subjects(X_subID_parts,selection)

# save the array with the Id's of each user per set, for future reference
if not os.path.exists('../data_division/'+dire):
    os.makedirs('../data_division/'+dire)
    print("made new directory: " + '../data_division/'+ dire )
    
np.save('../data_division/'+dire+'/train_subject_ID_.npy',subID)
np.save('../data_division/'+dire+'/validation_subject_ID_.npy',subID_val)
np.save('../data_division/'+dire+'/test_subject_ID_.npy',subID_test)

print ("generated division files in ../data_division/"+dire)
