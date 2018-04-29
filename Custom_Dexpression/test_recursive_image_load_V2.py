import numpy as np
import cv2
import scipy as sc
# import matplotlib.pyplot as plt
from glob import glob 
import re
import os

import tflearn
from face_detect import cutFace
from showNumpyInfo import showInfo

import logging
from memory_profiler import memory_usage

# if __name__ == "__main__":
# logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
#                     format="%(asctime)-15s %(levelname)-8s %(message)s")
# logging.info("hello")

def logprint(stre):
	print(stre)
	logging.info(stre)

#if print is 1 than for each loaded image stats are print
def load_CKP_data(datasetPath, printData = 0):
	
 	#stores the current working directory path to restore at the end and goes to the given database path
	curr_path = os.getcwd()
	os.chdir(datasetPath)
	#get a list of URLs to all the emotion file that can be found
	label_list = glob('./Emotion/*/*/*.txt')

	X_data = [] # list of input data = climax images of emotions
	Y_data = [] # list of output data = emotion expressed in image
	X_subID = [] # list with subject Id of each X_data element 

	i = 0 #count of total amount of instances


	#count per emotion
	N = 0 # Neutral
	A = 0 # Anger
	C = 0 # Contempt
	D = 0 # Disgust
	F = 0 # Fear
	H = 0 # Happy
	Sa = 0# Saddness
	Su = 0# Surprise

	#subject counter
	sub = 0
	lastSub = 0
	subjectID = 0

	#count of dimensions
	dimMap = {}

	
	# # image
	# img = cv2.imread(str)


	#iterate through the list of label files, open corresponding images
	for fn in label_list:

		str = './cohn-kanade-images'+ fn[9:-12] + '.png'
		

		# image
		# img = cv2.imread(str)
		# img = sc.misc.imread(str,flatten=True)
		# cv2.imshow('image',img)

		image = cv2.imread(str)
		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# cv2.imshow("Faces found", img)
		# cv2.waitKey(0)

		#sizeStr = repr(img.shape[0]) + "x" + repr(img.shape[1]) #+ "x" + repr(img.shape[2])

		# test accessing single elements
		# if sizeStr in dimMap:
		# 	dimMap[sizeStr] = dimMap[sizeStr] + 1 
		# else:
		# 	dimMap[sizeStr] = 1

		# read emotion labels from the file
		file = open(fn,'r')
		fileTxt = file.read()
		emotionNr = int(fileTxt[3:4])
		file.close()

		#determine the subject (number is within URL)
		subjectID = fn[11:14]
		if subjectID != lastSub :
			sub = sub + 1
		lastSub = subjectID 

		#determine the emotion of the image
		if(emotionNr == 0):
			emotion = 'Neutral'
			N = N + 1
		elif(emotionNr == 1):
			emotion = 'Anger'
			A = A + 1
		elif(emotionNr == 2):
			emotion = 'Contempt'
			C = C + 1
		elif(emotionNr == 3):
			emotion = 'Disgust'
			D = D + 1
		elif(emotionNr == 4):
			emotion = 'Fear '
			F = F + 1
		elif(emotionNr == 5):
			emotion = 'Happy'
			H = H + 1
		elif(emotionNr == 6):
			emotion = 'Saddness'
			Sa = Sa + 1
		else:
			emotion = 'Surprise'
			Su = Su + 1

		

	# store image and emotion label in the X_data and Y_data lists
		X_data.append(img)
		Y_data.append(emotionNr)
		X_subID.append(subjectID)

		
		i = i+1
		




		if printData :
			print("-------------------------------")

			print(str) #./cohn-kanade-images/S506/002/S506_004_00000038.png

			print("image type:   " + repr(type(img)))


			print('image ' + repr(i) + " " + emotion)

	#show stattistics
	logging.info("--------- Overal stattistics ---------  ")
	logging.info("amount of Subjects  : " + repr(sub))
	logging.info("amount of Instances : " + repr(i))
	logging.info("code = 1 = Anger      " + repr(A) + " instances: " + repr(np.round((float(A)/i)*100),decimals=2))
	logging.info("code = 2 = Neutral    " + repr(N) + " instances: " + repr(np.round((float(N)/i)*100),decimals=2))
	logging.info("code = 3 = Disgust    " + repr(D) + " instances: "+ repr(np.round((float(D)/i)*100),decimals=2))
	logging.info("code = 4 = Fear       " + repr(F) + " instances: "+ repr(np.round((float(F)/i)*100),decimals=2))
	logging.info("code = 5 = Happy      " + repr(H) + " instances: "+ repr(np.round((float(H)/i)*100),decimals=2))
	logging.info("code = 6 = Saddness   " + repr(Sa) +" instances: "+ repr(np.round((float(Sa)/i)*100),decimals=2))
	logging.info("code = 7 = Surprise   " + repr(Su) +" instances: "+ repr(np.round((float(Su)/i)*100),decimals=2))
	logging.info("--------- dimensions ---------  ")

	logging.info("--------- last elements in lists ---------  ")
	logging.info("length X_data" + repr(len(X_data))) 
	logging.info("length Y_data" + repr(len(Y_data))) 

	# cv2.imshow('image',X_data[len(X_data)-1])
	plt.figure('last image')
	plt.imshow(X_data[len(X_data)-1], cmap='gray')#, interpolation='nearest');
	plt.show()

	print("length Y_data " + repr(Y_data[len(Y_data)-1])) 
	print("image type " + repr(type(Y_data[len(Y_data)-1]))) 

   

	cv2.waitKey(0)
		

	cv2.destroyAllWindows()

	os.chdir(curr_path)

	return [X_data,Y_data,X_subID]



def load_formated_data(datasetPath, printData = 0, cascPath = "G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/DeXpression-master_chris/haarcascade.xml"
 ,allData = False,neutral = False):

	logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
	                    format="%(asctime)-15s %(levelname)-8s %(message)s")
	logging.info("hello")


	if(allData==True):
		data = load_all_annotated_CKP_data(datasetPath,printData)
	elif(neutral == True):
		data = load_all_annotated_CKP_data_neutral(datasetPath,printData)
	else:		
		data = load_CKP_data(datasetPath,printData)

	# save part of the data in temp.npy files
	amount_sets = len(data)//50

	if not os.path.exists('./temp'):
		os.makedirs('./temp')
		print("made new directory: " + './temp')

	start = 0    
	for i in range(0,amount_sets-1):
		stre = "./temp/temp_" + repr(i) + "_" + repr(start) + "-" + repr(start+49) + ".npy"
		npy.save(stre,data[start:start+50])
		start = start +50

	stre = "temp_" + 	repr(i) + "_" + repr(start) + "-" + repr(len(data)) + ".npy"
	npy.save(stre,data[start:])

	mem_usage = memory_usage(-1, interval=.2, timeout=1)
	str = 'Maximum memory usage before data = null: ' + repr( mem_usage)
	logprint(str)

	data = None;

	mem_usage = memory_usage(-1, interval=.2, timeout=1)
	str = 'Maximum memory usage after data = null: ' + repr( mem_usage)
	logprint(str)



	images = data[0]

	# cut faces from each images and resize (downsample) to input dimension
	X_gray = []
	X_test = []

	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)
	stre ="total amount of images is "+ repr( len(images) )
	logprint(stre)
	logprint("start cutting the faces")

	list_files = glob("./temp/*")

	for file in list_files:
		logprint("load file: " + file)
		images = np.load(file)

		for i in range(0,len(images)):	   
			cut_img = cutFace(images[i],224,224,faceCascade)
			X_gray = np.append(X_gray,cut_img)
			if(i%50 == 0):
				stre =" "+ repr(i) + " images have been processed"
				print(stre)
				logging.info(stre)
				#monitor the memory usage of the current proces for 1 second, sample each 0.2 s return list of MB values
				mem_usage = memory_usage(-1, interval=.2, timeout=1)
				str = 'Maximum memory usage: ' + repr( mem_usage)
				logprint(str)

			if printData==1 :
				cv2.imshow("example", cut_img.reshape((224,224)))
				cv2.waitKey(0)

	X_gray = X_gray.reshape([-1,224,224,1])
	print(X_gray.shape)

	# reformat the emotionlabels to create a vector of 7 zeros only one of these elements is set to 1 indicating the emotion label
	labels = []

	logprint("start coverting Y_class_data to output data")
	for lab in data[1]:
	    inst = np.zeros(7)
	    inst[lab-1] = 1
	    labels = np.append(labels,inst)
	    
	print("size Y 2: " + repr(labels.shape))
	    
	labels = labels.reshape([-1,7]) 

	print("size labels flatten: " + repr(len(labels)))
	print("size labels shape: " + repr(labels.shape))
	print("type labels: " + repr(type(labels)))

	outputData = [X_gray,labels,data[2]]

	logprint("the end")

	return outputData

#create .npy files with only the 
def create_formated_data(datasetPath, printData = 0, cascPath = "G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/DeXpression-master_chris/haarcascade.xml"):
	data = load_formated_data(datasetPath, printData, cascPath)
	
	if not os.path.exists('../data/CKP'):
		os.makedirs('../data/CKP')
		print("made new directory: " + ' ../data/CKP')

	np.save('../data/CKP/X.npy',data[0])
	np.save('../data/CKP/Y.npy',data[1])
	np.save('../data/CKP/subjectIDs.npy',data[2])

def create_all_CKP_formated_data(datasetPath, printData = 0, cascPath = "G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/DeXpression-master_chris/haarcascade.xml"):
	data = load_formated_data(datasetPath, printData, cascPath, allData = True)

	if not os.path.exists('../data/CKP_all'):
		os.makedirs('../data/CKP_all')
		print("made new directory: " + ' ../data/CKP_all')

	np.save('../data/CKP_all/X.npy',data[0])
	np.save('../data/CKP_all/Y.npy',data[1])
	np.save('../data/CKP_all/subjectIDs.npy',data[2])

def create_all_CKP_formated_data_neutral(datasetPath, printData = 0, cascPath = "G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/DeXpression-master_chris/haarcascade.xml"):
	data = load_formated_data(datasetPath, printData, cascPath, neutral = True)

	if not os.path.exists('../data/CKP_all_neutral'):
		os.makedirs('../data/CKP_all_neutral')
		print("made new directory: " + ' ../data/CKP_all_neutral')

	np.save('../data/CKP_all_neutral/X.npy',data[0])
	np.save('../data/CKP_all_neutral/Y.npy',data[1])
	np.save('../data/CKP_all_neutral/subjectIDs.npy',data[2])

dataPath = 'G:/Documenten/personal/school/MaNaMA_AI/thesis/databases/wikipedia_list/cohn-Kanade/CK+'

# uncomment following function to the formatted database
# create_formated_data(dataPath)



# split dataset in parts based the subject IDs
# IN:
# dataset = [X,Y,subjectIDs] (list)
# NrParts = the number of parts the dataset is split
# OUT:
# parts = [X_parts, Y_parts, X_subID_parts] =>  list of Lists containing containing 'NrParts' amount of np.ndarrays
def split_dataset(dataset, NrParts , logfile = "logfile"):

	logging.basicConfig(level=logging.DEBUG, filename = logfile, filemode="a+",
	                format="%(asctime)-15s %(levelname)-8s %(message)s")
	logging.info("start logging to " + logfile)

	X = dataset[0]
	Y = dataset[1]
	subjectIDs = dataset[2]

	# indices = np.argsort(subjectIDs,kind='mergesort',axis = 0)

	# X = X[indices]
	# Y = Y[indices]
	# subjectIDs = subjectIDs[indices] 

	partSize = (np.floor(len((subjectIDs))/(NrParts))).astype('uint8')
	logprint("partsize "  + repr(partSize))


	lastSubID = 0 #the ID of the subject of the last instance
	partCount = 0 #The amount of instances in this part already
	partCountList= [] #the amount of instaces per part in list format
	partID = 0    #keeps the number of the part that is being filled right now

	X_array = np.asarray([])
	Y_array = np.asarray([])
	sub_array = np.asarray([])

	X_parts = []
	Y_parts = []
	X_subID_parts = []  

	print("amount of instances " + repr(len(subjectIDs)))
	#iterate through al the ID's if a part is full and the subjectID is different from the previous, fill the next part 
	for i in range(0,len(subjectIDs)):

	#     print("this part IDs top border ", partSize ," and partCount = ", partCount)
	    if((partCount >= partSize) & (lastSubID != subjectIDs[i])):

	        logprint('part[' + repr(partID) + '] has ' + repr(partCount) + ' instances')
	        logprint('function is at instance ' + repr(i))
	        partCountList.append(partCount)
	        
	        #fill the parts to the partslists
	        X_parts.append(X_array)
	        Y_parts.append(Y_array)
	        X_subID_parts.append(sub_array)

	        #refresh the temporary arrays
	        X_array = np.asarray([])
	        Y_array = np.asarray([])
	        sub_array = np.asarray([])
	        #reset counter & set next part to be filled
	        partCount = 0
	        partID = partID + 1
	        
	    X_array = np.append(X_array ,X[i])
	    Y_array = np.append(Y_array ,Y[i])
	    sub_array = np.append(sub_array ,subjectIDs[i])  

	    partCount = partCount +1
	    lastSubID = subjectIDs[i]
	  
	if(partCount > 1):
	    logprint('part['+ repr(partID) + '] has' + repr(partCount) +  ' instances')
	    logprint('function is at instance ' + repr(i))
	    partCountList.append(partCount)

	    #fill the parts to the parts lists
	    X_parts.append(X_array)
	    Y_parts.append(Y_array)
	    X_subID_parts.append(sub_array)

	logprint('#parts in X_parts '+ repr(len(X_parts)))
	logprint('#parts in Y_parts ' + repr(len(Y_parts)))
	logprint('#parts in Y_subID_parts ' + repr(len(X_subID_parts)))
	logprint("Count per part " + repr( partCountList))
	logprint("There were " + repr( len(subjectIDs)) +  "instances, the parts contain " + repr( sum(partCountList)) +"instances" )


	return [X_parts,Y_parts,X_subID_parts]

def split_subject_dataset(subjectIDs, NrParts, logfile="logfile"):

	logging.basicConfig(level=logging.DEBUG, filename = logfile, filemode="a+",
	                format="%(asctime)-15s %(levelname)-8s %(message)s")
	logging.info("start logging to " + logfile)

	div = len(subjectIDs)/(NrParts)
	floor = np.floor(div)
	partSize =  floor

	logprint("DEBUG div "+ repr(div) + " floor "+ repr(floor)  + " partSize " +repr( partSize))

	##partSize = (np.floor(len(subjectIDs)/(NrParts))).astype('uint8')
	logprint("subjectIDs length "+ repr(len(subjectIDs)) + " divide by "+ repr( NrParts) + " gives a partsize  " + repr(partSize))

	lastSubID = 0 #the ID of the subject of the last instance
	partCount = 0 #The amount of instances in this part already
	partCountList= [] #the amount of instaces per part in list format
	partID = 0    #keeps the number of the part that is being filled right now

	sub_array = np.asarray([])

	X_subID_parts = []  

	logprint("amount of instances " + repr(len(subjectIDs)))
	#iterate through al the ID's if a part is full and the subjectID is different from the previous, fill the next part 
	for i in range(0,len(subjectIDs)):

	#     print("this part IDs top border ", partSize ," and partCount = ", partCount)
	    if((partCount >= partSize) & (lastSubID != subjectIDs[i])):

	        logprint('part[' + repr(partID) + '] has ' + repr(partCount) + ' instances')
	        logprint('function is at instance ' + repr(i))
	        partCountList.append(partCount)
	        
	        #fill the parts to the partslists
	        X_subID_parts.append(sub_array)

	        #refresh the temporary arrays
	        sub_array = np.asarray([])
	        #reset counter & set next part to be filled
	        partCount = 0
	        partID = partID + 1
	        
	    sub_array = np.append(sub_array ,subjectIDs[i])  

	    partCount = partCount +1
	    lastSubID = subjectIDs[i]
	  
	if(partCount > 1):
	    logprint('part[' + repr(partID) + '] has ' + repr(partCount) + ' instances')
	    logprint('function is at instance ' + repr(i))
	    partCountList.append(partCount)

	    #fill the parts to the parts lists
	    X_subID_parts.append(sub_array)

	logprint('#parts in Y_subID_parts ' + repr(len(X_subID_parts)))
	logprint("Count per part " + repr( partCountList))
	logprint("There were " + repr(len(subjectIDs)) + "instances, the parts contain " + repr(sum(partCountList)) + "instances" )


	return X_subID_parts

def split_subject_dataset_better(subjectIDs, NrParts, logfile="logfile"):

	logging.basicConfig(level=logging.DEBUG, filename = logfile, filemode="a+",
	                format="%(asctime)-15s %(levelname)-8s %(message)s")
	logging.info("start logging to " + logfile)

	div = len(subjectIDs)/(NrParts)
	floor = np.floor(div)
	partSize =  floor

	logprint("DEBUG div "+ repr(div) + " floor "+ repr(floor)  + " partSize " +repr( partSize))

	##partSize = (np.floor(len(subjectIDs)/(NrParts))).astype('uint8')
	logprint("subjectIDs length "+ repr(len(subjectIDs)) + " divide by "+ repr( NrParts) + " gives a partsize  " + repr(partSize))

	lastSubID = 0 #the ID of the subject of the last instance
	partCount = 0 #The amount of instances in this part already
	partCountList= [] #the amount of instaces per part in list format
	partID = 0    #keeps the number of the part that is being filled right now

	sub_array = np.asarray([])

	X_subID_parts = []  

	logprint("amount of instances " + repr(len(subjectIDs)))
	#iterate through al the ID's if a part is full and the subjectID is different from the previous, fill the next part 
	for i in range(0,len(subjectIDs)):

	#     print("this part IDs top border ", partSize ," and partCount = ", partCount)
	    if((partCount >= partSize) & (lastSubID != subjectIDs[i])):

	        logprint('part[' + repr(partID) + '] has ' + repr(partCount) + ' instances')
	        logprint('function is at instance ' + repr(i))
	        partCountList.append(partCount)
	        
	        #fill the parts to the partslists
	        X_subID_parts.append(sub_array)

	        #refresh the temporary arrays
	        sub_array = np.asarray([])
	        #reset counter & set next part to be filled
	        partCount = 0
	        partID = partID + 1
	        
	    sub_array = np.append(sub_array ,subjectIDs[i])  

	    partCount = partCount +1
	    lastSubID = subjectIDs[i]
	  
	if(partCount > 1):
	    logprint('part[' + repr(partID) + '] has ' + repr(partCount) + ' instances')
	    logprint('function is at instance ' + repr(i))
	    partCountList.append(partCount)

	    #fill the parts to the parts lists
	    X_subID_parts.append(sub_array)

	logprint('#parts in Y_subID_parts ' + repr(len(X_subID_parts)))
	logprint("Count per part " + repr( partCountList))
	logprint("There were " + repr(len(subjectIDs)) + "instances, the parts contain " + repr(sum(partCountList)) + "instances" )


	return X_subID_parts


# divide the subjects from a database according to the division in selection
# IN:
# X_subID_parts = the subject labels of the data set split in to N parts
# selection = a list with 3 lists [select select_val select_test]
# OUT:
# A list of 3 lists with subject numbers for training, validation and test sets
def divide_subjects(X_subID_parts,selection, logfile = "logfile"):
	select =  selection[0]
	select_val =  selection[1]
	select_test=  selection[2]

	logging.basicConfig(level=logging.DEBUG, filename= logfile, filemode="a+",
	                format="%(asctime)-15s %(levelname)-8s %(message)s")
	logging.info("hello")


	selected_partNr = len(select )+ len(select_val )+len(select_test )
	logprint("number of X_subID_parts" + repr(len(X_subID_parts)))
	logprint("number of selected parts" + repr( selected_partNr))
	if(len(X_subID_parts) < selected_partNr):
		logprint("there are LESS parts then there are selected parts")
	elif(len(X_subID_parts) > selected_partNr):
		logprint("there are MORE parts then there are selected parts")
	else :
		logprint("Equal parts and selected parts")


	subID = np.asarray([])
	subID_val = np.asarray([])
	subID_test = np.asarray([])

	# create the trainings set based on the select array
	for i in select:
	    subID = np.append(subID,X_subID_parts[i])
	for i in select_val:    
	    subID_val = np.append(subID_val,X_subID_parts[i])
	for i in select_test:
	    subID_test = np.append(subID_test,X_subID_parts[i])

	subID = subID.astype('uint8')    
	subID_val = subID_val.astype('uint8')
	subID_test = subID_test .astype('uint8') 

	return [subID, subID_val, subID_test]
 
def printInfo(i, text):   
	if(i%50 == 0):
		logprint(text)
		stre =" "+ repr(i) + " images have been processed"
		logprint(stre)
		#monitor the memory usage of the current proces for 1 second, sample each 0.2 s return list of MB values
		mem_usage = memory_usage(-1, interval=.2, timeout=1)
		str = 'Maximum memory usage: ' + repr( mem_usage)
		logprint(str)



# divides the data in training, validation and test sets according to lists of the subjectIDs already divided over the 3
# IN:
# data = contain 3 1D-arrays: x,y and subject data [X_data, Y_data,X_subID]
# subIDs = contain 3 1D-arrays with the subject numbers for each set: train,val,test [subID subID_val subID_test]
# OUT:
# list of 6 arrays [X,Y,X_val,Y_val,X_test,Y_test]
def divide_data_to_subject(data,subIDs):
    X_data = data[0]
    Y_data = data[1]
    X_subID = data[2]
    
    subID     = np.unique(subIDs[0])
    subID_val = np.unique(subIDs[1])
    subID_test= np.unique(subIDs[2])
    
    X = np.asarray([])
    Y = np.asarray([]) 
    X_val = np.asarray([])
    Y_val = np.asarray([])
    X_test = np.asarray([])
    Y_test = np.asarray([])

#     extract images and labels belonging to the trainings list of subject IDs in subID
    for i in subID :
        for j in np.argwhere(X_subID==i):
#             print(j[0])
            X = np.append(X,X_data[j[0]])
            Y = np.append(Y,Y_data[j[0]])
        printInfo(i,"in trainings set out of "+ repr(len(subID_test)))
#     extract images and labels belonging to the validation list of subject IDs in subID_val            
    for i in subID_val :
        for j in np.argwhere(X_subID==i):
#             print(j[0])
            X_val  = np.append(X_val,X_data[j[0]])
            Y_val  = np.append(Y_val,Y_data[j[0]]) 
        printInfo(i,"in validation set out of "+ repr(len(subID_test)))
 #     extract images and labels belonging to the test list of subject IDs in subID_test           
    for i in subID_test :
        for j in np.argwhere(X_subID==i):
#             print(j[0])
            X_test = np.append(X_test,X_data[j[0]])
            Y_test = np.append(Y_test,Y_data[j[0]])
        printInfo(i,"in test set out of "+ repr(len(subID_test)))
            
    X = (X.reshape(-1,224,224,1)).astype('uint8')
    Y = (Y.reshape(-1,7)).astype('uint8')

    # create the validation set X_val and Y-val (SubID_val is not given to the network)
    X_val = X_val.reshape(-1,224,224,1).astype('uint8')
    Y_val = Y_val.reshape(-1,7).astype('uint8')

    # create the test set X_test and Y_test (SubID_test is not given to the network)
    X_test = X_test.reshape(-1,224,224,1).astype('uint8')
    Y_test = Y_test.reshape(-1,7).astype('uint8')

    return [X,Y,X_val,Y_val,X_test,Y_test]

# A more efficient version of the above
# divides the data in training, validation and test sets according to lists of the subjectIDs already divided over the 3
# IN:
# data = contain 3 1D-arrays: x,y and subject data [X_data, Y_data,X_subID]
# subIDs = contain 3 1D-arrays with the subject numbers for each set: train,val,test [subID subID_val subID_test]
# OUT:
# list of 6 arrays [X,Y,X_val,Y_val,X_test,Y_test]
def opti_divide_data_to_subject(data,subIDs):
    X_data = data[0]
    Y_data = data[1]
    X_subID = data[2]
    
    subID     = np.unique(subIDs[0])
    subID_val = np.unique(subIDs[1])
    subID_test= np.unique(subIDs[2])
    
    X = np.zeros((subIDs[0].shape[0]+10,224,224,1))
    Y = np.zeros((subIDs[0].shape[0]+10,7))
    X_val = np.zeros((subIDs[1].shape[0]+10,224,224,1))
    Y_val = np.zeros((subIDs[1].shape[0]+10,7))
    X_test = np.zeros((subIDs[2].shape[0]+10,224,224,1))
    Y_test = np.zeros((subIDs[2].shape[0]+10,7))

    index = 0
    index_val = 0
    index_test = 0

    showInfo(subID)
    showInfo(subID_val)
    showInfo(subID_test)
#     extract images and labels belonging to the trainings list of subject IDs in subID
    cumul = 0
    count = 0
    for i in subID :
        count = count + 1
        print("subID count", count)
        print("subID ", i)
        same =np.argwhere(X_subID==i)
        print("same ",same)
        print("same ",same.shape[0])
        cumul = cumul + same.shape[0]
        print("same cumul",cumul)
        for j in same:
            print(j[0])
            X[index] = X_data[j[0]]
            Y[index] = Y_data[j[0]]
            index =  index + 1
    print("index is ", index)

#     extract images and labels belonging to the validation list of subject IDs in subID_val            
    for i in subID_val :
        for j in np.argwhere(X_subID==i):
#             print(j[0])
            X_val[index_val]  = X_data[j[0]]
            Y_val[index_val]  = Y_data[j[0]]
            index_val =  index_val + 1
    print ("index_val is ", index_val)
 #     extract images and labels belonging to the test list of subject IDs in subID_test           
    for i in subID_test :
        for j in np.argwhere(X_subID==i):
#             print(j[0])
            X_test[index_test]  = X_data[j[0]]
            Y_test[index_test]  = Y_data[j[0]]
            index_test =  index_test + 1
    print("index_test is ", index_test)
            
    X = (X.reshape(-1,224,224,1)).astype('uint8')
    Y = (Y.reshape(-1,7)).astype('uint8')

    # create the validation set X_val and Y-val (SubID_val is not given to the network)
    X_val = X_val.reshape(-1,224,224,1).astype('uint8')
    Y_val = Y_val.reshape(-1,7).astype('uint8')

    # create the test set X_test and Y_test (SubID_test is not given to the network)
    X_test = X_test.reshape(-1,224,224,1).astype('uint8')
    Y_test = Y_test.reshape(-1,7).astype('uint8')

    return [X,Y,X_val,Y_val,X_test,Y_test]

def load_all_annotated_CKP_data(datasetPath, printData = 0):
	
 	#stores the current working directory path to restore at the end and goes to the given database path
	curr_path = os.getcwd()
	os.chdir(datasetPath)
	#get a list of URLs to all the emotion file that can be found
	label_list = glob('./Emotion/*/*/*.txt')

	X_data = [] # list of input data = climax images of emotions
	Y_data = [] # list of output data = emotion expressed in image
	X_subID = [] # list with subject Id of each X_data element 

	# each emotion clip has only a few relevant images that represent an emotion 
	relevant_part = 0.33 # the "precentage" of the end of the clip that has relevant images for that emotion


	i = 0 #count of total amount of instance
	tot_img_count = 0 # Count of how many images there are loaded

	#count per emotion
	N = 0 # Neutral
	A = 0 # Anger
	C = 0 # Contempt
	D = 0 # Disgust
	F = 0 # Fear
	H = 0 # Happy
	Sa = 0# Saddness
	Su = 0# Surprise

	#subject counter
	sub = 0
	lastSub = 0
	subjectID = 0

	#count of dimensions
	dimMap = {}

	
	# # image
	# img = cv2.imread(str)

	print("begin loading data")
	#iterate through the list of label files, open corresponding images
	for fn in label_list:

		str = './cohn-kanade-images'+ fn[9:-29] + '*.png'
		
		# print(fn[9:-29])
		# print(str)

		img_list = glob(str)

		# image
		# img = cv2.imread(str)
		# img = sc.misc.imread(str,flatten=True)
		# cv2.imshow('image',img)

		img = []

		for url in img_list:
			# print(url)
			image = cv2.imread(url)
			img.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

		# print(len(img))
		relevant_amount = int(round(len(img)*relevant_part))
		# print ("from ", len(img), "amount of images only ",relevant_amount, " is relevant")
		tot_img_count = tot_img_count + relevant_amount 
		# print("tot_img_count ", tot_img_count)

		# cv2.imshow("Faces found", img)
		# cv2.waitKey(0)

		#sizeStr = repr(img.shape[0]) + "x" + repr(img.shape[1]) #+ "x" + repr(img.shape[2])

		# test accessing single elements
		# if sizeStr in dimMap:
		# 	dimMap[sizeStr] = dimMap[sizeStr] + 1 
		# else:
		# 	dimMap[sizeStr] = 1

		# read emotion labels from the file
		file = open(fn,'r')
		fileTxt = file.read()
		emotionNr = int(fileTxt[3:4])
		file.close()

		#determine the subject (number is within URL)
		subjectID = fn[11:14]
		if subjectID != lastSub :
			sub = sub + 1
		lastSub = subjectID 

		#determine the emotion of the image
		if(emotionNr == 0):
			emotion = 'Neutral'
			N = N + relevant_amount 
		elif(emotionNr == 1):
			emotion = 'Anger'
			A = A + relevant_amount 
		elif(emotionNr == 2):
			emotion = 'Contempt'
			C = C + relevant_amount 
		elif(emotionNr == 3):
			emotion = 'Disgust'
			D = D + relevant_amount 
		elif(emotionNr == 4):
			emotion = 'Fear '
			F = F + relevant_amount 
		elif(emotionNr == 5):
			emotion = 'Happy'
			H = H + relevant_amount 
		elif(emotionNr == 6):
			emotion = 'Saddness'
			Sa = Sa + relevant_amount 
		else:
			emotion = 'Surprise'
			Su = Su + relevant_amount 

		

	# store image and emotion label in the X_data and Y_data lists
		for j in img[(len(img)-relevant_amount) :len(img)]:
			X_data.append(j)
			Y_data.append(emotionNr)
			X_subID.append(subjectID)

		# print("X_data length ",len(X_data))
		
		i = i+relevant_amount + 3
		
		if (sub%20 ==0):
			print(sub,"subjects loaded")

		if printData :
			print("-------------------------------")

			print(str) #./cohn-kanade-images/S506/002/S506_004_00000038.png
			print("amount of image added " , len(img))
			print("image type:   " + repr(type(img)))


			print('image ' + repr(i) + " " + emotion)

	#show stattistics
	print("--------- Overal stattistics ---------  ")
	print("amount of Subjects  : %d" % sub)
	print("amount of Instances : %d" % i)
	print("code = 0 = Neutral    %d instances: %.2f" % (N,((float(N)/i)*100)))
	print("code = 1 = Anger      %d instances: %.2f"  % (A,((float(A)/i)*100)))
	print("code = 2 = Contempt   %d instances: %.2f"  % (C,((float(C)/i)*100)))
	print("code = 3 = Disgust    %d instances: %.2f"  % (D,((float(D)/i)*100)))
	print("code = 4 = Fear       %d instances: %.2f"  % (F,((float(F)/i)*100)))
	print("code = 5 = Happy      %d instances: %.2f"  % (H,((float(H)/i)*100)))
	print("code = 6 = Saddness   %d instances: %.2f"  % (Sa,((float(Sa)/i)*100)))
	print("code = 7 = Surprise   %d instances: %.2f"  % (Su,((float(Su)/i)*100)))
	print("--------- dimensions ---------  ")
	# print(repr(dimMap))

	print("--------- last elements in lists ---------  ")
	print("length X_data " + repr(len(X_data))) 
	print("length Y_data " + repr(len(Y_data)))
	print("length X_subID " + repr(len(X_subID))) 
 

	# cv2.imshow('image',X_data[len(X_data)-1])
	plt.figure('last image')
	plt.imshow(X_data[len(X_data)-1], cmap='gray')#, interpolation='nearest');
	plt.show()

	print("length Y_data " + repr(Y_data[len(Y_data)-1])) 
	print("image type " + repr(type(Y_data[len(Y_data)-1]))) 

   

	cv2.waitKey(0)
		

	cv2.destroyAllWindows()

	os.chdir(curr_path)

	return [X_data,Y_data,X_subID]

datasetPath = 'G:/Documenten/personal/school/MaNaMA_AI/thesis/databases/wikipedia_list/cohn-Kanade/CK+'
# load_all_annotated_CKP_data(datasetPath, printData = 0)

# this function searched for CKP data X,Y and SubID in the absolute path if given
# if absolute path is not given is will go down in the directories depth deep searching for the /data directory
# IN:
# 	depth = the amount of directories the function wil go down to look for the /data directory
# 	absolute_path = the absolute path to the /data/ folder where the .npy example: C:\Documenten\data
def load_npy_files(depth = 5,direc = "CKP",absolute_path=None):

	w = "[WARNING]"
	s = "[SUCCEED]"
	f = "[FAILED]"

	

	if (absolute_path is not None):
		full_path = absolute_path +'/'+ direc 
		# Load data from: https://drive.google.com/drive/folders/1YWT8DJivNOZzQRPCiHDPY0LL_dymdQIS?usp=sharing
		#if not in working directory look in appointed directory
		try:
			X_data = np.load(full_path +  '/X.npy')
			Y_data = np.load(full_path + '/Y.npy')
			X_subID = (np.load(full_path + '/subjectIds.npy').astype('uint8'))
			print(s + " data found in absolute_path: ", full_path)

			showInfo(X_data,"X_data")
			showInfo(Y_data,"Y_data")
			showInfo(X_subID,"X_subID")


			return [X_data, Y_data, X_subID]
		except Exception as e:
			print(w + 'could not load from absolute_path: %s' , 'because of: %s'%e, 'error')

	path = "./data/"+ direc 

	for i in range(0,depth):
		try:

			X_data = np.load(path + '/X.npy')
			Y_data = np.load(path + '/Y.npy')
			X_subID = (np.load(path + '/subjectIDs.npy')).astype('uint8')
			print(s + "data found in path: ", path )

			showInfo(X_data,"X_data")
			showInfo(Y_data,"Y_data")
			showInfo(X_subID,"X_subID")

			return [X_data, Y_data, X_subID]
		except Exception as e:
			print(w + ' could not load from path: %s' %path,'\n because of: %s' %e, 'error')
		
		path = './.' + path

	print(f + "After searching 5 layer of directories data is still not found")


	return None

# if __name__=='__main__':
# 	load_npy_CKP_files()


#load 1/3 of the frames as emotional and 3 starting frames of each video as neutral
def load_all_annotated_CKP_data_neutral(datasetPath, printData = 0):
	

	logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")
	logging.info("hello")


 	#stores the current working directory path to restore at the end and goes to the given database path
	curr_path = os.getcwd()
	os.chdir(datasetPath)
	#get a list of URLs to all the emotion file that can be found
	label_list = glob('./Emotion/*/*/*.txt')

	X_data = [] # list of input data = climax images of emotions
	Y_data = [] # list of output data = emotion expressed in image
	X_subID = [] # list with subject Id of each X_data element 

	# each emotion clip has only a few relevant images that represent an emotion 
	relevant_part = 0.33 # the "precentage" of the end of the clip that has relevant images for that emotion


	i = 0 #count of total amount of instance
	tot_img_count = 0 # Count of how many images there are loaded

	#count per emotion
	N = 0 # Neutral
	A = 0 # Anger
	C = 0 # Contempt
	D = 0 # Disgust
	F = 0 # Fear
	H = 0 # Happy
	Sa = 0# Saddness
	Su = 0# Surprise

	#subject counter
	sub = 0
	lastSub = 0
	subjectID = 0

	#count of dimensions
	dimMap = {}

	
	# # image
	# img = cv2.imread(str)

	print("begin loading data")
	#iterate through the list of label files, open corresponding images
	for fn in label_list:

		str = './cohn-kanade-images'+ fn[9:-29] + '*.png'
		
		# print(fn[9:-29])
		# print(str)

		img_list = glob(str)

		# image
		# img = cv2.imread(str)
		# img = sc.misc.imread(str,flatten=True)
		# cv2.imshow('image',img)

		img = []
		img_neutral =[]

		relevant_amount = int(round(len(img_list)*relevant_part))
		
		# stre = "from " + repr( len(img_list)) +  "amount of images only " + repr(relevant_amount) + " is relevant"
		# logging.info(stre)
		tot_img_count = tot_img_count + relevant_amount 
		# print("tot_img_count ", tot_img_count)


		#the first 3 images are taken as being neutral
		count = 0
		for url in img_list[0:3]:
			# print(url)
			image = cv2.imread(url)
			img_neutral.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
			N = N + 1
			# stre = "neutral img = " + repr(count)
			# cv2.imshow(stre, image)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

			count = count +1

		# relevant amount of  the last frames are taken as belonging to that emotion
		count = 0
		for url in img_list[(len(img_list)-relevant_amount) :len(img_list)]:
			# print(url)
			image = cv2.imread(url)
			img.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

			# stre = "emotion img = " + repr(count)
			# cv2.imshow(stre, image)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			count = count +1


		#sizeStr = repr(img.shape[0]) + "x" + repr(img.shape[1]) #+ "x" + repr(img.shape[2])

		# test accessing single elements
		# if sizeStr in dimMap:
		# 	dimMap[sizeStr] = dimMap[sizeStr] + 1 
		# else:
		# 	dimMap[sizeStr] = 1

		# read emotion labels from the file
		file = open(fn,'r')
		fileTxt = file.read()
		emotionNr = int(fileTxt[3:4])
		file.close()

		#determine the subject (number is within URL)
		subjectID = fn[11:14]
		if subjectID != lastSub :
			sub = sub + 1
		lastSub = subjectID 

		#determine the emotion of the image
		if(emotionNr == 0):
			emotion = 'Neutral'
			N = N + relevant_amount
		elif(emotionNr == 1):
			emotion = 'Anger'
			A = A + relevant_amount 
		# elif(emotionNr == 2):
		# 	emotion = 'Contempt'
		# 	C = C + relevant_amount 
		elif(emotionNr == 3):
			emotion = 'Disgust'
			D = D + relevant_amount 
		elif(emotionNr == 4):
			emotion = 'Fear '
			F = F + relevant_amount 
		elif(emotionNr == 5):
			emotion = 'Happy'
			H = H + relevant_amount 
		elif(emotionNr == 6):
			emotion = 'Saddness'
			Sa = Sa + relevant_amount 
		else:
			emotion = 'Surprise'
			Su = Su + relevant_amount 



		if(emotionNr != 2):
			# store images of the neutral expression and emotion label in the X_data and Y_data lists
			for j in img_neutral:
				X_data.append(j)
				Y_data.append(2)
				X_subID.append(subjectID)	

			# store image and emotion label in the X_data and Y_data lists
			for j in img:
				X_data.append(j)
				Y_data.append(emotionNr)
				X_subID.append(subjectID)

			# print("X_data length ",len(X_data))
			
			i = i+len(img) + len(img_neutral)
		
		if (sub%20 ==0):
			logprint(repr(sub)+"subjects loaded")

		if printData :
			logging.info("-------------------------------")

			logging.info(str) #./cohn-kanade-images/S506/002/S506_004_00000038.png
			logging.info("amount of image added " + repr( len(img)))
			logging.info("image type:   " + repr(type(img)))


			logging.info('image ' + repr(i) + " " + emotion)

	#show stattistics
	logging.info("--------- Overal stattistics ---------  ")
	logging.info("amount of Subjects  : " + repr(sub))
	logging.info("amount of Instances : " + repr(i))
	logging.info("code = 1 = Anger      " + repr(A) + " instances: " + repr(np.round((float(A)/i)*100),decimals=2))
	logging.info("code = 2 = Neutral    " + repr(N) + " instances: " + repr(np.round((float(N)/i)*100),decimals=2))
	logging.info("code = 3 = Disgust    " + repr(D) + " instances: "+ repr(np.round((float(D)/i)*100),decimals=2))
	logging.info("code = 4 = Fear       " + repr(F) + " instances: "+ repr(np.round((float(F)/i)*100),decimals=2))
	logging.info("code = 5 = Happy      " + repr(H) + " instances: "+ repr(np.round((float(H)/i)*100),decimals=2))
	logging.info("code = 6 = Saddness   " + repr(Sa) +" instances: "+ repr(np.round((float(Sa)/i)*100),decimals=2))
	logging.info("code = 7 = Surprise   " + repr(Su) +" instances: "+ repr(np.round((float(Su)/i)*100),decimals=2))
	logging.info("--------- dimensions ---------  ")


	logging.info("--------- last elements in lists ---------  ")
	logging.info("length X_data " + repr(len(X_data))) 
	logging.info("length Y_data " + repr(len(Y_data)))
	logging.info("length X_subID " + repr(len(X_subID))) 
 

	# # cv2.imshow('image',X_data[len(X_data)-1])
	# plt.figure('last image')
	# plt.imshow(X_data[len(X_data)-1], cmap='gray')#, interpolation='nearest');
	# plt.show()

	print("length Y_data " + repr(Y_data[len(Y_data)-1])) 
	print("image type " + repr(type(Y_data[len(Y_data)-1]))) 



	os.chdir(curr_path)

	return [X_data,Y_data,X_subID]