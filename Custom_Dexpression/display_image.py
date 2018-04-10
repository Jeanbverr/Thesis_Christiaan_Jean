
def display(img,title = 'example',condor=False):
	if(condor is False):
		import cv2
		cv2.imshow(title, img.reshape((224,224)))
		cv2.waitKey(0)