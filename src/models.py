""" Contains a collection of different models implemented with keras. """
import argparse
import sys

import keras
import keras.applications as app
import keras.layers as l


def resnet50(shape, classes):
    """ ResNet50 """
    initial_model = app.resnet50.ResNet50(
        include_top=False,
        pooling='avg',
        weights='imagenet',
        input_shape=shape)
    last = initial_model.output
    preds = l.Dense(classes, activation='softmax')(last)
    model = keras.models.Model(initial_model.input, preds)
    model.name = "resnet50"
    return model

def mobilnet(shape, classes):
    """ MobilNet """
    initial_model = app.mobilenet.MobileNet(
        include_top=False,
        pooling='avg',
        weights='imagenet',
        input_shape=shape)
    last = initial_model.output
    preds = l.Dense(classes, activation='softmax')(last)
    model = keras.models.Model(initial_model.input, preds)
    model.name = "mobilnet"
    return model

def inception_resnet_v2(shape, classes):
    """ Inception-ResNet v2 """
    initial_model = app.inception_resnet_v2.InceptionResNetV2(
        include_top=False,
        pooling='avg',
        weights='imagenet',
        input_shape=shape)
    last = initial_model.output
    preds = l.Dense(classes, activation='softmax')(last)
    model = keras.models.Model(initial_model.input, preds)
    model.name = "inception_resnet_v2"
    return model

def xception(shape, classes):
    """ Xception """
    initial_model = app.xception.Xception(
        include_top=False,
        pooling='avg',
        weights='imagenet',
        input_shape=shape)
    last = initial_model.output
    preds = l.Dense(classes, activation='softmax')(last)
    model = keras.models.Model(initial_model.input, preds)
    model.name = "xception"
    return model

def densenet201(shape, classes):
    """ DenseNet201 """
    initial_model = app.DenseNet201(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=shape)
    last = initial_model.output
    preds = l.Dense(classes, activation='softmax')(last)
    model = keras.models.Model(initial_model.input, preds)
    model.name = "densenet201"
    return model

def densenet121(shape, classes):
    """ DenseNet121 """
    initial_model = app.DenseNet121(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=shape)
    last = initial_model.output
    preds = l.Dense(classes, activation='softmax')(last)
    model = keras.models.Model(initial_model.input, preds)
    model.name = "densenet121"
    return model

def densenet169(shape, classes):
    """ DenseNet169 """
    initial_model = app.DenseNet169(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=shape)
    last = initial_model.output
    preds = l.Dense(classes, activation='softmax')(last)
    model = keras.models.Model(initial_model.input, preds)
    model.name = "densenet169"
    return model

def vgg16(shape, classes):
    """ Vgg16 """
    initial_model = app.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=shape)
    preds = l.Flatten()(initial_model.output)
    x = l.Dense(1024)(preds)
    x = l.LeakyReLU(0.1)(x)
    x = l.Dropout(0.5)(x)
    x = l.Dense(classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[initial_model.input], outputs=[x])
    model.name = 'VGG16'
    return model

def vgg19(shape, classes):
    """ Vgg19 """
    initial_model = app.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=shape)
    preds = l.Flatten()(initial_model.output)
    x = l.Dense(1024)(preds)
    x = l.LeakyReLU(0.1)(x)
    x = l.Dropout(0.5)(x)
    x = l.Dense(classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[initial_model.input], outputs=[x])
    model.name = 'VGG19'
    return model

def inception_v3(shape,classes):
    """ inceptino v3 """
    initial_model = app.inception_v3.InceptionV3(
        include_top=False,
        weights='imagenet',
        pooling='avg',
        input_shape=shape)
    last = initial_model.output
    preds = l.Dense(classes, activation='softmax')(last)
    model = keras.models.Model(initial_model.input, preds)
    model.name = "inception_v3"
    return model

def handle_args(**kwargs):
    """ Handles the command line inputs. """
    shape = (256,256,3)
    classes = 10000
    try:
        model = getattr(sys.modules[__name__], kwargs['modelname'].lower())(shape, classes)
        model.summary()
    except AttributeError:
        print("{} has not been implemented yet.".format(kwargs['modelname']))

def main():
    """ Entry poin """
    parser = argparse.ArgumentParser()
    parser.add_argument('modelname', metavar='MODEL_NAME')
    args = parser.parse_args()
    handle_args(**vars(args))


if __name__ == '__main__':
    main()
