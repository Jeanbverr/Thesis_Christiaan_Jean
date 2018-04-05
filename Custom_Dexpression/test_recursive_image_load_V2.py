import numpy as np
import cv2
import scipy as sc
import matplotlib.pyplot as plt
from glob import glob 
import re
import os

import tflearn
from face_detect import cutFace


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

			print("---")
			print(repr(img.item((300,300))))


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
	print("length X_data" + repr(len(X_data))) 
	print("length Y_data" + repr(len(Y_data))) 

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
 ):

	data = load_CKP_data(datasetPath,printData)


	images = data[0]

	# cut faces from each images and resize (downsample) to input dimension
	X_gray = []
	X_test = []

	# Create the haar cascade
	faceCascade = cv2.CascadeClassifier(cascPath)

	for i in range(0,len(images)):	   
		cut_img = cutFace(images[i],224,224,faceCascade)
		X_gray = np.append(X_gray,cut_img)
		if printData==1 :
			cv2.imshow("example", cut_img.reshape((224,224)))
			cv2.waitKey(0)

	X_gray = X_gray.reshape([-1,224,224,1])
	print(X_gray.shape)

	# reformat the emotionlabels to create a vector of 7 zeros only one of these elements is set to 1 indicating the emotion label
	labels = []

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

	return outputData


def create_formated_data(datasetPath, printData = 0, cascPath = "G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/DeXpression-master_chris/haarcascade.xml"):
	data = load_formated_data(datasetPath, printData, cascPath)
	np.save('CKP_X.npy',data[0])
	np.save('CKP_Y.npy',data[1])
	np.save('CKP_subjectIDs.npy',data[2])

dataPath = 'G:/Documenten/personal/school/MaNaMA_AI/thesis/databases/wikipedia_list/cohn-Kanade/CK+'

# uncomment following function to the formatted database
# create_formated_data(dataPath)



# split dataset in parts based the subject IDs
# IN:
# dataset = [X,Y,subjectIDs] (list)
# NrParts = the number of parts the dataset is split
# OUT:
# parts = [X_parts, Y_parts, X_subID_parts] =>  list of Lists containing containing 'NrParts' amount of np.ndarrays
def split_dataset(dataset, NrParts):
	
	X = dataset[0]
	Y = dataset[1]
	subjectIDs = dataset[2]

	partSize = (np.floor(len((subjectIDs))/(NrParts+1))).astype('uint8')
	print("partsize " , partSize)


	lastSubID = 0 #the ID of the subject of the last instance
	partCount = 0 #The amount of instances in this part already
	partID = 0    #keeps the number of the part that is being filled right now

	X_array = np.asarray([])
	Y_array = np.asarray([])
	sub_array = np.asarray([])

	X_parts = []
	Y_parts = []
	X_subID_parts = []  

	#iterate through al the ID's if a part is full and the subjectID is different from the previous, fill the next part 
	for i in range(0,len(subjectIDs)):

	#     print("this part IDs top border ", partSize ," and partCount = ", partCount)
		if((partCount >= partSize) & (lastSubID != subjectIDs[i])):

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

	print('#parts in X_parts ',len(X_parts))
	print('#parts in Y_parts ',len(Y_parts))
	print('#parts in Y_subID_parts ',len(X_subID_parts))

	return [X_parts,Y_parts,X_subID_parts]

# divide the subjects from a database according to the division in selection
# IN:
# X_subID_parts = the subject labels of the data set split in to N parts
# selection = a list with 3 lists [select select_val select_test]
# OUT:
# A list of 3 lists with subject numbers for training, validation and test sets
def divide_subjects(X_subID_parts,selection):
    select =  selection[0]
    select_val =  selection[1]
    select_test=  selection[2]

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
#     extract images and labels belonging to the validation list of subject IDs in subID_val            
    for i in subID_val :
        for j in np.argwhere(X_subID==i):
#             print(j[0])
            X_val  = np.append(X_val,X_data[j[0]])
            Y_val  = np.append(Y_val,Y_data[j[0]]) 
 #     extract images and labels belonging to the test list of subject IDs in subID_test           
    for i in subID_test :
        for j in np.argwhere(X_subID==i):
#             print(j[0])
            X_test = np.append(X_test,X_data[j[0]])
            Y_test = np.append(Y_test,Y_data[j[0]])
            
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
	tot_img_count = 0
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

		str = './cohn-kanade-images'+ fn[9:-29] + '*.png'
		
		# print(fn[9:-29])
		print(str)

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

		print(len(img))
		tot_img_count = tot_img_count + len(img)
		print("tot_img_count ", tot_img_count)

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
		print("X_data length ",len(X_data))
		
		i = i+1
		




		if printData :
			print("-------------------------------")

			print(str) #./cohn-kanade-images/S506/002/S506_004_00000038.png

			print("image type:   " + repr(type(img)))

			print("---")
			print(repr(img.item((300,300))))


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
	print("length X_data" + repr(len(X_data))) 
	print("length Y_data" + repr(len(Y_data))) 

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