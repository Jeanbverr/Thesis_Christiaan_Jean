import argparse
import numpy as np

# Author Christiaan Vanbergen
# This python script compares the subject division over train,validation and test sets
# in the local data_division/ directory to those in the directory given in the --URL argument as a global URL

# usage example
# python compare_division.py --URL G:\Documenten\dexpression\data_division

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--URL',type=str, help="increase output verbosity",default = 0) 

args = parser.parse_args()
print("URL = " , args.URL)
URL = args.URL

#if in windows change the \ to /
URL = URL.replace("\\",'/')

print("corrected URL = " , URL)
print("------------------------------------")
#load local the subject distribution over the different datasets
subID = (np.load('data_division/train_subject_ID.npy')).astype('uint8')
subID_val = (np.load('data_division/validation_subject_ID.npy')).astype('uint8')
subID_test = (np.load('data_division/test_subject_ID.npy')).astype('uint8')
subIDs = [subID, subID_val, subID_test]

#load URL the subject distribution over the different datasets
subID_2 = (np.load(URL + '/train_subject_ID.npy')).astype('uint8')
subID_val_2 = (np.load(URL + '/validation_subject_ID.npy')).astype('uint8')
subID_test_2 = (np.load(URL + '/test_subject_ID.npy')).astype('uint8')
subIDs_2 = [subID_2, subID_val_2, subID_test_2]

if(subID.shape == subID_2.shape and subID_val.shape == subID_val_2.shape and subID_test.shape == subID_test_2.shape ):
	print("shape_test: SUCCEED: sets are the same shape")
else:
	print("shape_test: FAILED : sets are NOT the same shape")

if(subID.all() == subID_2.all() and subID_val.all() == subID_val_2.all() and subID_test.all() == subID_2.all() ):
	print("equality_test: SUCCEED: sets are completely the same")
else :
	print("equality_test: FAILED : sets are NOT the same")