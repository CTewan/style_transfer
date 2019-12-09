import cv2
import numpy as np

def _read_image(path):
	img = cv2.imread(path)

	return img

def _resize_image(img, shape=None):
	if shape == None:
		shape = (128, 128)
	
	img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_LINEAR)

	return img

def _expand_dimensions(img):
	img = np.expand_dims(img, axis=0)

	return img

def _normalize_image(img):
	img = img.astype(dtype="float32")
	img = cv2.normalize(src=img, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

	return img

def preprocess_image(path, shape=None):
	img = _read_image(path)
	img = resize_image(img, shape=shape)
	img = expand_dimensions(img)
	img = normalize_image(img)

	return img
