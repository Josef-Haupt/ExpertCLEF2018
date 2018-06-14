import numpy as np
import train
import os
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
import config

path = "/datasets/ExpertCLEF2017/ExpertCLEF2018/t2"
modelpath = "/datasets/ExpertCLEF2017/models/densenet201/run16/model.h5"

gen = ImageDataGenerator(rescale=1./255)
gen = gen.flow_from_directory(
    path,
    target_size=(256, 256),
    class_mode=None,
    shuffle=False)

model = keras.models.load_model(modelpath, custom_objects={"top3_acc": train.top3_acc})

p = model.predict_generator(gen)
d = [ np.argmax(el) for el in p ]
l = sorted(os.listdir(config.MAIN_DATASET_PATH))
pass
