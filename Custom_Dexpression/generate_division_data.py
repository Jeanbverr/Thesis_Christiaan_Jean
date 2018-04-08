import numpy as np

from test_recursive_image_load_V2 import  split_subject_dataset
from test_recursive_image_load_V2 import  divide_subjects

# Author: Christiaan Vanbergen
# This script generates a division of subjects of a dataset to be later used to divide the input and output data
# over train,validation and test data.
# To generate a custom division change the inputs of split_dataset and the select_* variables
# generate_data.py generates the CKP_X.npy, CKP_Y.npy,CKP_subjectIDs.npy loaded here
# Or the files can be found on: https://drive.google.com/drive/folders/1YWT8DJivNOZzQRPCiHDPY0LL_dymdQIS?usp=sharing

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--add',type=str,
                   help='string to be added to output name')

args = parser.parse_args()
# print(args.ID)
stre = args.add



try:
    # X_data = np.load('../data/CKP_X.npy')
    # Y_data = np.load('../data/CKP_Y.npy')
    # X_subID = (np.load('../data/CKP_subjectIds.npy')).astype('uint8')

    X_data = np.load('../data/CKP_X_all.npy')
    Y_data = np.load('../data/CKP_Y_all.npy')
    X_subID = (np.load('../data/CKP_subjectIDs_all.npy')).astype('uint8')
except:
	#if data is not found in usual /data folder, load from custom global path
    # X_data = np.load('G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/github/Thesis_Christiaan_Jean/data/CKP_X.npy')
    # Y_data = np.load('G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/github/Thesis_Christiaan_Jean/data/CKP_Y.npy')
    # X_subID = (np.load('G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/github/Thesis_Christiaan_Jean/data/CKP_subjectIds.npy')).astype('uint8')
    X_data = np.load('G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/github/Thesis_Christiaan_Jean/data/CKP_X_all.npy')
    Y_data = np.load('G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/github/Thesis_Christiaan_Jean/data/CKP_Y_all.npy')
    X_subID = (np.load('G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/github/Thesis_Christiaan_Jean/data/CKP_subjectIDs_all.npy')).astype('uint8')



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
if (stre is None):
	np.save('data_division/train_subject_ID.npy',subID)
	np.save('data_division/validation_subject_ID.npy',subID_val)
	np.save('data_division/test_subject_ID.npy',subID_test)
else:
	np.save('data_division/train_subject_ID_'+stre+'.npy',subID)
	np.save('data_division/validation_subject_ID_'+stre+'.npy',subID_val)
	np.save('data_division/test_subject_ID_'+stre+'.npy',subID_test)	
