import argparse
import numpy as np

# Author Christiaan Vanbergen
# This python script compares the subject division over train,validation and test sets
# in the local data_division/ directory to those in the directory given in the --URL argument as a global URL

# usage example
# python compare_division.py --URL G:\Documenten\dexpression\data_division

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('URL',type=str, help="path to the /data_division/ directory of the files to compare, example G:\Documenten\Custom_Dexpression\data_division",default = 0) 
parser.add_argument('dir',type=str, help="the name of the directory within data_division that has to be compared, example: CKP",default = 0) 
args = parser.parse_args()

print("URL = " , args.URL)
print("dir = " , args.dir)
URL = args.URL
dire= args.dir

#if in windows change the \ to /
URL = URL.replace("\\",'/')

path1 = '../data_division/'+dire
path2 = URL + '/'+dire

print("corrected URL = " , URL)
print("------------------------------------")
#load local the subject distribution over the different datasets
subID = (np.load(path1 +'/train_subject_ID.npy')).astype('uint8')
subID_val = (np.load(path1 +'/validation_subject_ID.npy')).astype('uint8')
subID_test = (np.load(path1 +'/test_subject_ID.npy')).astype('uint8')
subIDs = [subID, subID_val, subID_test]

#load URL the subject distribution over the different datasets
subID_2 = (np.load(path2 +'/train_subject_ID.npy')).astype('uint8')
subID_val_2 = (np.load(path2 +'/validation_subject_ID.npy')).astype('uint8')
subID_test_2 = (np.load(path2 +'/test_subject_ID.npy')).astype('uint8')
subIDs_2 = [subID_2, subID_val_2, subID_test_2]


print("comparing files in \n" + path1 + "\nTo files in\n"+ path2 + "\n------THE RESULTS-----")
if(subID.shape == subID_2.shape and subID_val.shape == subID_val_2.shape and subID_test.shape == subID_test_2.shape ):
	print("shape_test: SUCCEED: sets are the same shape")
else:
	print("shape_test: FAILED : sets are NOT the same shape")

if(subID.all() == subID_2.all() and subID_val.all() == subID_val_2.all() and subID_test.all() == subID_2.all() ):
	print("equality_test: SUCCEED: sets are completely the same")
else :
	print("equality_test: FAILED : sets are NOT the same")