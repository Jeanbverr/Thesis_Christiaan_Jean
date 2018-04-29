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
from test_recursive_image_load_V2 import split_subject_dataset
from test_recursive_image_load_V2 import divide_subjects
from test_recursive_image_load_V2 import divide_data_to_subject
from test_recursive_image_load_V2 import load_npy_files

import logging
from memory_profiler import memory_usage

def generate_predivided_data():
	#create the folders and loggin files

	load_dir = 'CKP' #the directory that is created in the predivided_data directory


	new = False
	if not os.path.exists('./predivided_data/'+load_dir):
	    os.makedirs('./predivided_data/'+load_dir)
	    new = True
	 

	# configure the logging    
	logfile = './predivided_data/'+load_dir+'/division_log.txt'

	if os.path.exists(logfile):
		os.remove(logfile)

	logging.basicConfig(level=logging.DEBUG, filename = logfile, filemode="a+",
	                    format="%(asctime)-15s %(levelname)-8s %(message)s")
	logging.info("start logging to " + logfile)

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
		logprint("Name " + repr(VarName))
		logprint("type: " + repr(type(Var)))
		logprint("Dtype: " + repr(Var.dtype))
		logprint("shape: " + repr(Var.shape))

	if(new == True):
		logprint("made new directory: " + '../predivided_data/'+ load_dir )


	# load the data
	data= load_npy_files(5,load_dir)
	[X_data,Y_data,X_subID] = data

	#check the sizes of all the data for future reference
	logprint("CHECK data type and shape")

	logprint("Type of data var "+ repr(type(data)))
	logprint("Type of data[0] "+ repr( type(data[0])))

	showInfo(X_data,"X_data")
	showInfo(Y_data,"Y_data")
	showInfo(X_subID ,"X_subID")
	logprint(X_subID)
	logprint("-----DONE DEBUG---------")

	#sort data to subjects
	logprint("start sort data to subjects -----------------------")

	indices = np.argsort(X_subID,kind='mergesort',axis = 0)

	X_data = X_data[indices]
	Y_data = Y_data[indices]
	X_subID = X_subID[indices]


	logprint("CHECK SORTED data type and shape")

	logprint("Type of data var "+ repr(type(data)))
	logprint("Type of data[0] "+ repr( type(data[0])))

	showInfo(X_data,"X_data")
	showInfo(Y_data,"Y_data")
	showInfo(X_subID ,"X_subID")
	logprint(X_subID)
	logprint("-----DONE DEBUG---------")

	#END sort data to subjects
	logprint("end sort data to subjects -----------------------")

	#start splitting the dataset (code previously housed in  generate_division_data)
	logprint("start splitting the dataset (code previously housed in  generate_division_data")
	# seperate the dataset in 10 parts that do not share users: 
	#   6 parts train set
	#   3 part validation set
	#   1 part test set
	# allows future possibility for 10-fold crossover

	X_subID_parts = split_subject_dataset(X_subID,10,logfile)

	# data = split_dataset(data,10,logfile)

	# [X_data,Y_data,X_subID] = data

	select =  [0,5,2,7,4,8] # 60%
	select_val = [6,3,9] 	# 30%
	select_test= [1]		# 10%

	selection = [select, select_val, select_test]

	lenSelect = len(select)
	lenSelect_val = len(select_val)
	lenSelect_test = len(select_test)
	lenTot = lenSelect + lenSelect_val + lenSelect_test
	lenParts = len(X_subID_parts)

	if(lenTot != lenParts):
		logprint("Warning it is possible that not all parts of the dataset are used")

	[subID, subID_val, subID_test] =  divide_subjects(X_subID_parts,selection, logfile)
	subIDs = [subID, subID_val, subID_test]

	logprint("there are " + repr(lenParts) + " parts")
	logprint("the training   set contains parts " + repr(select) + " that is " + repr(lenSelect) + " of those parts which is " + repr((lenSelect/lenTot)*100) + "%")
	logprint("the validation set contains parts " + repr(select_val) + " that is " + repr(lenSelect_val) + " of those parts which is " + repr((lenSelect_val/lenTot)*100) + "%")
	logprint("the test       set contains parts " + repr(select_test) + " that is " + repr(lenSelect_test) + " of those parts which is " + repr((lenSelect_test/lenTot)*100) + "%")

	logprint("end splitting the dataset ------------------------------------------")
	# end splitting the dataset ------------------------------------------


	# load the premade division data
	# try:
	# 	subID = (np.load('./data_division/'+load_dir+'/train_subject_ID.npy')).astype('uint8')
	# 	subID_val = (np.load('./data_division/'+load_dir+'/validation_subject_ID.npy')).astype('uint8')
	# 	subID_test = (np.load('./data_division/'+load_dir+'/test_subject_ID.npy')).astype('uint8')
	# 	subIDs = [subID, subID_val, subID_test]
	# except:
	# 	subID = (np.load('../data_division/'+load_dir+'/train_subject_ID.npy')).astype('uint8')
	# 	subID_val = (np.load('../data_division/'+load_dir+'/validation_subject_ID.npy')).astype('uint8')
	# 	subID_test = (np.load('../data_division/'+load_dir+'/test_subject_ID.npy')).astype('uint8')
	# 	subIDs = [subID, subID_val, subID_test]

	logprint("CHECK subID type and shape")
	logprint("Type of data var ", type(data))
	logprint("Type of data[0] ",  type(data[0]))
	logprint("Type of subIDs var ", type(subIDs))
	logprint("Type of subIDs[0] ",  type(subIDs[0]))
	showInfo(subID,"subID")
	logprint(subID)
	showInfo(subID_val,"subID_val")
	logprint(subID_val)
	showInfo(subID_test ,"subID_test")
	logprint(subID_test)
	logprint('the sum of instances in all the subID arrays', subID.shape[0]+subID_val.shape[0]+subID_test.shape[0] ,' needs to be the same as the first dimension of the X_data ', X_data.shape[0]  )
	logprint("-----DONE DEBUG---------")

	#monitor the memory usage of the current proces for 1 second, sample each 0.2 s return list of MB values
	mem_usage = memory_usage(-1, interval=.2, timeout=1)
	logprint('Maximum memory usage: %s' % mem_usage)


	# divide the data in the predivided sets and store them in the given load_dir
	logprint("dividing the data")
	divided_data = divide_data_to_subject(data,subIDs)




	X = (divided_data[0].reshape(-1,224,224,1)).astype('uint8')
	Y = (divided_data[1].reshape(-1,7)).astype('uint8')

	# create the validation set X_val and Y-val (SubID_val is not given to the network)
	X_val = divided_data[2].reshape(-1,224,224,1).astype('uint8')
	Y_val = divided_data[3].reshape(-1,7).astype('uint8')

	# create the test set X_test and Y_test (SubID_test is not given to the network)
	X_test = divided_data[4].reshape(-1,224,224,1).astype('uint8')
	Y_test = divided_data[5].reshape(-1,7).astype('uint8')

	showInfo(X,"X")
	showInfo(Y,"Y")
	logprint('--------------------------')
	showInfo(X_val,"X_val")
	showInfo(Y_val,"Y_val")
	logprint('--------------------------')
	showInfo(X_test,"X_test")
	showInfo(Y_test,"Y_test")


	np.save('./predivided_data/'+load_dir+'/X_train.npy',X)
	np.save('./predivided_data/'+load_dir+'/Y_train.npy',Y)

	np.save('./predivided_data/'+load_dir+'/X_val.npy',X_val)
	np.save('./predivided_data/'+load_dir+'/Y_val.npy',Y_val)

	np.save('./predivided_data/'+load_dir+'/X_test.npy',X_test)
	np.save('./predivided_data/'+load_dir+'/Y_test.npy',Y_test)

	# for i in range(0,len(X)):
	# 	cv2.imshow("img "+repr(i),X[i])
	# 	cv2.waitKey(0)
	# 	if(i%10 ==0):
	# 		cv2.destroyAllWindows()


if __name__=='__main__':
	generate_predivided_data()