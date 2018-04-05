import os
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file

pretrained_model = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
modhash = '89f56a39a78454e96379348bddd78c0d'


def get_args():
	print("test")
	parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
												 "and estimates age and gender for the detected faces.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--weight_file", type=str, default=None,
						help="path to weight file (e.g. weights.18-4.06.hdf5)")
	parser.add_argument("--depth", type=int, default=16,
						help="depth of network")
	parser.add_argument("--width", type=int, default=8,
						help="width of network")
	parser.add_argument(
        '-f',
        '--file',
        help='Path for input file. First line should contain number of lines to search in'
    )
	args = parser.parse_args("AAA --file /path/to/sequences.txt".split())

	return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
	size = cv2.getTextSize(label, font, font_scale, thickness)[0]
	x, y = point
	cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
	cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def main(depth = 16, k = 8, weight_file = None):
	if not weight_file:
		weight_file = get_file("weights.18-4.06.hdf5", pretrained_model, cache_subdir="pretrained_models",
							   file_hash=modhash, cache_dir=os.path.dirname(os.path.abspath(__file__)))
	print("still nothing3")
	# for face detection
	detector = dlib.get_frontal_face_detector()

	# load model and weights
	img_size = 64
	model = WideResNet(img_size, depth=depth, k=k)()
	model.load_weights(weight_file)

	print("running main")
	img = os.path.abspath("frames/S040-001.jpg")
	#img = os.path.abspath("faces/Sessions/2894/S019-113.jpg")
	input_img = cv2.imread(img, 1)
	# cv2.imshow("img",input_img)
	img_h, img_w, _ = np.shape(input_img)
	detected = detector(input_img, 1)
	print("detected: ",detected)
	if(len(detected) == 0):
		input_img = np.rot90(input_img, k = 3)
	faces = np.empty((len(detected), img_size, img_size, 3))
	print("faces: ", faces)
	if len(detected) > 0:
		for i, d in enumerate(detected):
			x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
			xw1 = max(int(x1 - 0.4 * w), 0)
			yw1 = max(int(y1 - 0.4 * h), 0)
			xw2 = min(int(x2 + 0.4 * w), img_w - 1)
			yw2 = min(int(y2 + 0.4 * h), img_h - 1)
			cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
			# cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
			faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

		# predict ages and genders of the detected faces
		results = model.predict(faces)
		predicted_genders = results[0]
		ages = np.arange(0, 101).reshape(101, 1)
		predicted_ages = results[1].dot(ages).flatten()

		# draw results
		for i, d in enumerate(detected):
			label = "{}, {}".format(int(predicted_ages[i]),
									"F" if predicted_genders[i][0] > 0.5 else "M")
			draw_label(input_img, (d.left(), d.top()), label)

	# input_img = cv2.resize(input_img, dsize=tuple([s // 2 for s in image.shape if s > 3])[::-1])
	print(predicted_ages)
	if(img_h>2000):
		new_h = int(img_h/3)
		new_w = int(img_w/3)
		input_img = cv2.resize(input_img, (new_w, new_h))
	cv2.imshow("result", input_img)
	key = cv2.waitKey()
if __name__ == '__main__':
	print("test")
	main()
