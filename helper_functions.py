import os

import cv2
import numpy as np

from tensorflow.keras.applications import (
        vgg19
    )

##############################################################################
#                             Get User input                                 #
##############################################################################

def get_content_and_style_image_path():
    content_img_path = input(
            "Enter path for content image "
            "(Press 'Enter' to use default): "
        )

    content_img_path = content_img_path.strip()

    style_img_path = input(
            "Enter path for style image "
            "(Press 'Enter' to use default): "
        )

    style_img_path = style_img_path.strip()

    if content_img_path == "":
        content_img_path = "Data/content_1.jpeg"

    if style_img_path == "":
        style_img_path = "Data/style_1.jpeg"

    while not os.path.isfile(content_img_path):
        content_img_path = input(
                "Please enter a valid path for content image "
                "(Press 'Enter' to use default): "
            )

        content_img_path = content_img_path.strip()

    while not os.path.isfile(style_img_path):
        style_img_path = input(
                "Enter path for style image "
                "(Press 'Enter' to use default): "
            )

        style_img_path = style_img_path.strip()
    return content_img_path, style_img_path

def _check_valid_tuple(output_dim):
    part_1 = output_dim[0] != "(" 
    part_2 = output_dim[-1] != ")" 
    part_3 = output_dim.count(",") != 1

    return part_1 and part_2 and part_3

def _check_dim_in_range(dim):
    if dim > 255 or dim < 128:
        return False

    return True

def _check_valid_dimensions(output_dim):
    dimensions = output_dim[1:-1]
    height = dimensions.split(",")[0].strip()
    width = dimensions.split(",")[1].strip()

    try:
        height = int(height)
        width = int(width)

        if _check_dim_in_range(height) and _check_dim_in_range(width):
            return True

        return False

    except:
        return False

def _convert_to_tuple(output_dim):
    dimensions = output_dim[1:-1]
    height = dimensions.split(",")[0].strip()
    width = dimensions.split(",")[1].strip()

    height = int(height)
    width = int(width)

    return (height, width)

def get_output_dim():
    output_dim = input(
            "Enter the desired shape for output image. "
            "(Press 'Enter' to use default). "
            "Example: (128, 128)"
        )
    output_dim = output_dim.strip()

    if output_dim == "":
        output_dim = (128, 128)

        return output_dim

    invalid_dimensions = True
    while invalid_dimensions:
        valid_tuple = _check_valid_tuple(output_dim)

        if valid_tuple:
            valid_dimensions = _check_valid_dimensions(output_dim)

            if valid_dimensions:
                break

        output_dim = input(
                "Please enter a valid output shape. "
                "(Press 'Enter' to use default). "
                "Example: (128, 128): "
            )
        output_dim = output_dim.strip()

    output_dim = convert_to_tuple(output_dim)
    return output_dim

def get_model_name():
    available_models = ["VGG19"]

    model_name = input(
            "Enter desired model to use. "
            "(Press 'Enter' to use default): "
            "Available: {}: ".format(available_models)         
        )
    model_name = model_name.strip()

    if model_name == "":
        model_name == "VGG19"

    while model_name not in available_models:
        model_name = input(
                "Please enter a valid model. "
                "(Press 'Enter to use default): "
                "Available: {}: ".format(available_models)
            )

        model_name = model_name.strip()

    return model_name

def get_style_weight():
    style_weight = input(
            "Enter style weight "
            "(Press 'Enter' to use default). "
            "Example: 0.25: "
        )

    style_weight = style_weight.strip()

    if style_weight == "":
        style_weight = 1.0

    invalid_input = True
    while invalid_input:
        try:
            style_weight = float(style_weight)
            break

        except:
            style_weight = input(
                    "Please enter a valid style weight "
                    "(Press 'Enter' to use default). "
                    "Example: 0.25: "
                )

            style_weight = style_weight.strip()

    return style_weight

def get_content_weight():
    content_weight = input(
            "Enter content weight "
            "(Press 'Enter' to use default): "
            "Example: 0.25"
        )

    content_weight = content_weight.strip()

    if content_weight == "":
        content_weight = 0.025

    invalid_input = True
    while invalid_input:
        try:
            content_weight = float(content_weight)
            break

        except:
            content_weight = input(
                    "Please enter a valid content weight "
                    "(Press 'Enter' to use default): "
                    "Example: 0.25"
                )

            content_weight = content_weight.strip()

    return content_weight

def get_variation_weight():
    variation_weight = input(
            "Enter variation weight "
            "(Press 'Enter' to use default): "
            "Example: 0.25"
        )

    variation_weight = variation_weight.strip()

    if variation_weight == "":
        variation_weight = 1.0

    invalid_input = True
    while invalid_input:
        try:
            variation_weight = float(variation_weight)
            break

        except:
            variation_weight = input(
                    "Enter variation weight "
                    "(Press 'Enter' to use default): "
                    "Example: 0.25"
                )

            variation_weight = variation_weight.strip()

    return variation_weight
##############################################################################
#                              Preprocess Image                              #
##############################################################################

def _read_image(img_path):
    # Reads image in BGR format
    img = cv2.imread(img_path)

    return img

def _resize_image(img, shape=None):
    if shape == None:
        shape = (128, 128)
    
    img = cv2.resize(img, dsize=shape, interpolation=cv2.INTER_LINEAR)

    return img

def _expand_dimensions(img):
    img = np.expand_dims(img, axis=0)

    return img


def _normalize_image(img, model_name="VGG19"):
    if model_name == "VGG19":
        img = vgg19.preprocess_input(img)

    return img

def preprocess_image(img_path, model_name="VGG19", shape=None):
    img = _read_image(img_path=img_path)
    img = _resize_image(img=img, shape=shape)
    img = _expand_dimensions(img=img)
    img = _normalize_image(img=img, model_name=model_name)

    return img

##############################################################################
#                               Deprocess Image                              #
##############################################################################
def deprocess_image(img, mean_val):
    img[:, :, 0] += mean_val[2]
    img[:, :, 1] += mean_val[1]
    img[:, :, 2] += mean_val[0]

    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astime("uint8")

    return img

##############################################################################
#                                  Save Image                                #
##############################################################################

def save_img(img_path, img):
    cv2.imwrite(img_path, img)