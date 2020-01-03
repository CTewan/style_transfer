
import numpy as np

from tensorflow.keras.applications import (
        vgg19
    )

import tensorflow.keras.backend as K

from tensorflow.keras.models import (
        load_model
    )

from scipy.optimize import fmin_l_bfgs_b

from helper_functions import (
        get_content_and_style_image_path,
        get_output_dim,
        get_model_name,
        get_style_weight,
        get_content_weight,
        get_variation_weight,
        preprocess_image,
        deprocess_image,
        save_img
    )

##############################################################################
#                             Model Related                                  #
##############################################################################
def _get_model(input_shape, model_name="VGG16"):
    if model_name == "VGG19":
        model = vgg19.VGG19(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet"
            )

    """
    elif model_name == "Xception":
        model = Xception(
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
    """
    return model

def _get_layers(model):
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    return layers

def _get_placeholder_img(shape):
    placeholder_img = K.placeholder(shape=shape, name="placeholder")

    return placeholder_img

def _get_model_input(content_img, style_img, output_img):
    ipt = K.concatenate([content_img, style_img, output_img], axis=0)

    return ipt

def _get_imagenet_mean():
    mean_val = [123.68, 116.779, 103.939]

    return mean_val

##############################################################################
#                                   Loss                                     #
##############################################################################

def _content_loss(content_img, output_img):
    content_loss = K.sum(K.square(output_img - content_img))

    return content_loss 

def _gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))

    return gram

def _style_loss(style_img, output_img, height, width, channels):
    size = height * width
    channels = channels

    input_style = _gram_matrix(style_img)
    output_style = _gram_matrix(output_img)

    style_loss_numerator = K.sum(K.square(input_style - output_style)) 
    style_loss_denominator = 4 * (channels ** 2) * (size **2)
    style_loss = style_loss_numerator / style_loss_denominator

    return style_loss

def _total_variation_loss(output_img, height, width):
    a = K.square(
            output_img[:, :height-1, :width-1, :] 
            - output_img[:, 1:, :width-1, :]
        )

    b = K.square(
            output_img[:, :height-1, :width-1, :]
            - output_img[:, :height-1, 1:, :]
        )

    variation_loss = K.sum(K.pow(a + b, 1.25))

    return variation_loss

def _update_content_loss(loss, layers, content_weight):
    layer = layers["block5_conv2"]
    content_img = layer[0, :, :, :]
    output_img = layer[2, :, :, :]
    content_loss = _content_loss(
            content_img=content_img,
            output_img=output_img
        )
    loss = loss + content_weight * content_loss

    return loss

def _update_style_loss(loss, layers, style_weight, height, width, channels):
    STYLE_LAYERS = ['block1_conv1', 'block2_conv1',
                      'block3_conv1', 'block4_conv1',
                      'block5_conv1']

    for layer_name in STYLE_LAYERS:
        layer_features = layers[layer_name]
        style_img = layer_features[1, :, :, :]
        output_img = layer_features[2, :, :, :]
        style_loss = _style_loss(
                style_img=style_img,
                output_img=output_img,
                height=height,
                width=width,
                channels=channels
            )

        loss = loss + (style_weight / len(STYLE_LAYERS)) * style_loss

    return loss

def _update_variation_loss(loss, output_img, variation_weight, height, width):
    variation_loss = _total_variation_loss(
            output_img=output_img,
            height=height,
            width=width
        )

    loss = loss + variation_weight * variation_loss

    return loss

def _evaluate_loss_and_grad(x, function, height, width, channels):
    x = x.reshape((1, height, width, channels))
    loss_grad = function([x])
    loss = loss_grad[0]

    if len(loss_grad[1:]) == 1:
        grad = loss_grad[1].flatten().astype("float64")

    else:
        grad = np.array(loss_grad[1:]).flatten().astype("float64")

    return loss, grad

class Evaluator(object):
    def __init__(self, height, width, channels, output_img, outputs):
        self.loss = None
        self.grad = None
        self.function = None
        self.height = height
        self.width = width
        self.channels = channels
        self.output_img = output_img
        self.outputs = outputs


    def get_loss(self, x):
        loss, grad = _evaluate_loss_and_grad(
                x=x,
                function=self.get_function(),
                height=self.height,
                width=self.width,
                channels=self.channels
            )

        self.loss = loss
        self.grad = grad

        return self.loss

    def get_grad(self, x):
        grad = np.copy(self.grad)

        self.loss = None
        self.grad = None

        return grad

    def get_function(self):
        # Given image, calculates loss and gradient
        self.function = K.function([self.output_img], self.outputs)

        return self.function

def _generate_image(evaluator, height, width, channels):
    ITERATIONS=20

    x = np.random.uniform(0, 255, (1, height, width, channels)) - 128
    #x = K.placeholder(shape=(1, height, width, channels))

    for i in range(ITERATIONS):
        x, loss, info = fmin_l_bfgs_b(
                evaluator.get_loss,
                x.flatten(),
                #x,
                fprime=evaluator.get_grad,
                maxfun=20)

        print("Loss for iteration {}: {}".format(i, loss))

        output = deprocess_image(x.copy())
        output_name = r"Output\style_img_iteration_{}.jpeg".format(i)

        save_img(img=output, img_path=output_name)

def main():
    # Get user inputs - image path, output shape, model
    content_img_path, style_img_path = get_content_and_style_image_path()
    output_dim = get_output_dim() # Default (128, 128)
    model_name = get_model_name()
    style_weight = get_style_weight()
    content_weight = get_content_weight()
    variation_weight = get_variation_weight()

    # Preprocess images (1, 128, 128, 3)
    content_img = preprocess_image(
            img_path=content_img_path,
            model_name=model_name,
            shape=output_dim
        )
    style_img = preprocess_image(
            img_path=style_img_path,
            model_name=model_name,
            shape=output_dim
        )

    height = content_img.shape[1]
    width = content_img.shape[2]
    channels = content_img.shape[3]
    input_shape = (height, width, channels)
    output_shape = content_img.shape
    output_img = _get_placeholder_img(shape=output_shape)
    
    # Prepare model
    imagenet_mean = _get_imagenet_mean()

    ipt = _get_model_input(
            content_img=content_img,
            style_img=style_img,
            output_img=output_img
        )
    model = _get_model(input_shape=input_shape, model_name=model_name)
    
    # Calculate loss
    layers = _get_layers(model=model)

    loss = K.variable(0.0)
    loss = _update_content_loss(
            loss=loss,
            layers=layers,
            content_weight=content_weight
        )
    loss = _update_style_loss(
            loss=loss,
            layers=layers,
            style_weight=style_weight,
            height=height,
            width=width,
            channels=channels
        )
    loss = _update_variation_loss(
            loss=loss,
            output_img=output_img,
            variation_weight=variation_weight,
            height=height,
            width=width
        )

    grads = K.gradients(loss, output_img)
    outputs = [loss]

    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    

    evaluator = Evaluator(
            height=height,
            width=width,
            channels=channels,
            output_img=output_img,
            outputs=outputs)

    _generate_image(
            #x=content_img,
            evaluator=evaluator,
            height=height,
            width=width,
            channels=channels
        )

if __name__ == "__main__":
    K.clear_session()
    main()

    

    



     