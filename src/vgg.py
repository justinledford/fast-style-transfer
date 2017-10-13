import numpy as np
import keras.backend as K
from keras.layers import Lambda, Input
from keras.applications.vgg19 import VGG19

MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])

def net(input_image):
    vgg = VGG19(include_top=False, input_tensor=Input(tensor=input_image))
    net = {layer.name: layer.output for layer in vgg.layers}
    return vgg, net

def preprocess(image):
    return Lambda(lambda x: x - MEAN_PIXEL)(image)


def unprocess(image):
    return Lambda(lambda x: x + MEAN_PIXEL)(image)
