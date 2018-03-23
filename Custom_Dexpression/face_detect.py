import cv2
import sys
import os
import numpy as np

# # Get user supplied values
# imagePath = os.path.abspath("test_img.png")
# cascPath = "G:/Documenten/personal/school/MaNaMA_AI/thesis/implementation/dexpression/github_1/DeXpression-master_chris/haarcascade.xml"

# # # # Create the haar cascade
# faceCascade = cv2.CascadeClassifier(cascPath)

# # # # Read the image
# image = cv2.imread(imagePath)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# print(type(gray))
# print(type(gray[0]))
# print(gray.dtype)
# print(gray.shape)

#give grayscale image and 2D height and width of output and a facecascade
def cutFace(img,width,height,casc):

	# Detect faces in the image
	faces = casc.detectMultiScale(
	    img,
	    scaleFactor=1.1,
	    minNeighbors=5,
	    minSize=(30, 30),
	    flags = cv2.CASCADE_SCALE_IMAGE
	)


	print("image shape X " + repr(img.shape[1]))
	print("image shape Y " + repr(img.shape[0]))

	# cv2.imshow("Faces found", img)

	# cv2.waitKey(0)
	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
	# 
	#     
	for (x, y, w, h) in faces:
		if w > h :
			he = w//2
			wi = w//2
		else :
			he = h//2
			wi = h//2

		x_m = (x+w//2)
		y_m = (y+h//2)

		# he = height//2
		# wi = width//2
		# cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
		#cut_image = img[(X_mid-(width//2)):(X_mid+(width//2)),(Y_mid-(height//2)):(Y_mid+(height//2))]
		cut_image = img[y_m-he:y_m+he,x_m-wi:x_m+wi]
		# print("x-co: ",x, "y-co: ",y, "width: ",w, "height: ",h)
		
		cut_image = cv2.resize(cut_image,(width,height), interpolation = cv2.INTER_AREA)

	print("image type " + repr(cut_image.dtype))
	print("image shape " + repr(cut_image.shape))

	return cut_image

# cut_image = cutFace(gray,224,224,faceCascade)

# cv2.imshow("Faces found", cut_image)

# cv2.waitKey(0)
