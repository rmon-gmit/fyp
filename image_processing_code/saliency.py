import math
import numpy as np

import tensorflow as tf

from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.models import Sequential, Model, Input
from keras.models import load_model

class Saliency:

    # Constructor - Not sure what this will take in yet...
    def __init__(self, dataset, model_path):
        self.dataset = dataset
        self.model_path = model_path
        self.model = self.__create_model()


    # Method to create model
    def __create_model(self):


        return model

    # Training method
    def train(self, model_path, epochs, steps):
        self.model.summary()

        self.model.fit(x=self.dataset.data_generator(),
                       epochs=epochs,
                       steps_per_epoch=steps,
                       max_queue_size=10,
                       workers=1,
                       verbose=1)

        self.model.save(model_path)


