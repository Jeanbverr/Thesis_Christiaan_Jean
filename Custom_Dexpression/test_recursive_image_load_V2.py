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
	#os.chdir('G:/Documenten/personal/school/MaNaMA_AI/thesis/databases/wikipedia_list/cohn-Kanade/CK+')
	os.chdir(datasetPath)
	label_list = glob('./Emotion/*/*/*.txt')

	X_data = [] # list of input data = climax images of emotions
	Y_data = [] # list of output data = emotion expressed in image

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

	#count of dimensions
	# dimMap = {'490x640x3': 0}
	dimMap = {}

	# str = './cohn-kanade-images'+ fn[9:-12] + '.png'
	# print(str) #./cohn-kanade-images/S506/002/S506_004_00000038.png

	# print(fn[11:14])

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

		# emotion label
		file = open(fn,'r')
		fileTxt = file.read()

		emotionNr = int(fileTxt[3:4])
		file.close()

		#determine the subject
		if fn[11:14] != lastSub :
			sub = sub + 1
		lastSub = fn[11:14] 

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

		
		i = i+1
		




		if printData :
			print("-------------------------------")

			print(str) #./cohn-kanade-images/S506/002/S506_004_00000038.png

			print("image type:   " + repr(type(img)))
			print("# dimensions: " + repr(img.ndim))
			# print("size X:       " + repr(img.shape[0]))
			# print("size Y:       " + repr(img.shape[1]))
			# print("size Z:       " + repr(img.shape[2]))
			# print("size:         " + sizeStr)

			print("---")
			print(repr(img.item((300,300))))
			# print(repr(img.item((300,300,0))))
			# print(repr(img.item((300,300,1))))
			# print(repr(img.item((300,300,2))))

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

	return [X_data,Y_data]

# load_CKP_data(1)

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

	X_gray = X_gray.reshape([-1,224,224,1])
	print(X_gray.shape)

	# reformat the emotionlabels to create a vector of 7 zeros only one of these elements is set to 1 indicating the emotion label
	labels = []

	for lab in data[1]:
	    inst = np.zeros(7)
	    inst[lab-1] = 1
	    #print (inst)
	    labels = np.append(labels,inst)
	    
	print("size Y 2: " + repr(labels.shape))
	    
	labels = labels.reshape([-1,7])

	print("size labels flatten: " + repr(len(labels)))
	print("size labels shape: " + repr(labels.shape))
	print("type labels: " + repr(type(labels)))

	outputData = [X_gray,labels]

	return outputData

dataPath = 'G:/Documenten/personal/school/MaNaMA_AI/thesis/databases/wikipedia_list/cohn-Kanade/CK+'
#data = load_formated_database(dataPath,0)

def create_formated_data(datasetPath, printData = 0, cascPath = "G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/DeXpression-master_chris/haarcascade.xml"):
	data = load_formated_data(datasetPath, printData, cascPath)
	np.save('CKP_X.npy',data[0])
	np.save('CKP_Y.npy',data[1])

# create_formated_data(dataPath)