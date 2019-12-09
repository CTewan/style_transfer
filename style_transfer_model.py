from tensorflow.keras.applications import (
		VGG16,
		Xception,
		Resnet,
		InceptionV3
	)

from tensorflow.keras.backend import (
		concatenate,
		placeholder,
		sum,
		square
	)

from tensorflow.keras.models import (
		load_model
	)

from helper_functions import (
		preprocess_image
	)

def _get_model(input_shape, model_name="VGG16"):
	if model_name == "VGG16":
		model = VGG16(
				input_shape=input_shape,
				include_top=False,
				weights="imagenet"
			)
	elif model_name == "Xception":
		model = Xception(
				input_shape=input_shape,
				include_top=False,
				weights="imagenet"
			)

	elif model_name == "Resnet":
		model = Resnet(
				input_shape=input_shape,
				include_top=False,
				weights="imagenet"
			)

	elif model_name == "InceptionV3":
		model = InceptionV3(
				input_shape=input_shape,
				include_top=False,
				weights="imagenet"
			)

	return model

def _get_content_and_style_image_path():
	content_img_path = input("Enter path for content image (Press 'Enter' if none): ")
	style_img_path = input("Enter path for style image (Press 'Enter' if none): ")

	if content_img_path is None:
		content_img_path = "Data/content_1.jpeg"

	if style_img_path is None:
		style_img_path = "Data/style_1.jpeg"

	return content_img_path, style_img_path

def _get_output_shape():
	output_shape = input("Enter the desired shape for output image (Press 'Enter' if none): ")

	if output_shape is None:
		output_shape = (128, 128)

	return output_shape

def _get_model_name():
	model_name = input("Enter desired model to use: ")

	return model_name

def _get_placeholder_img(shape):
	placeholder_img = placeholder(shape=shape)

	return placeholder_img

def _get_model_input(content_img, style_img, output_img):
	ipt = concatenate([content_img, style_img, output_img], axis=0)

	return ipt

def _content_loss(content_img, output_img):
	loss = sum(square(output_img - content_img))

	return loss

def style_loss(style_img, output_img):
	pass

def main():
	content_img_path, style_img_path = _get_content_and_style_image_path()
	output_shape = _get_output_shape()
	model_name = _get_model_name()

	content_img = preprocess_image(path=content_img_path, shape=output_shape)
	style_img = preprocess_image(path=style_img_path, shape=output_shape)
	output_shape = content_img.shape
	output_img = _get_placeholder_img(shape=output_shape)
	input_shape = (content_img.shape[1], content_img.shape[2], content_img.shape[3])

	ipt = _get_model_input(content_img, style_img, output_img)
	model = _get_model(input_shape=input_shape, model_name=model_name)




	 