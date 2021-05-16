"""
    Name: Ross Monaghan
    File: hopenet2.py
    Description: File containing second attempts at training hopenet model
    Date: 15/05/21
"""

import datasets
import tensorflow as tf
import numpy as np

from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.models import Model, Input

PROJECT_DIR = "C:/Users/rossm/OneDrive - GMIT/Year 4/Final Year Project/image_processing_code/"
MODELS_DIR = PROJECT_DIR + "models/"
AFLW2000_DATA_DIR = "C:/Users/rossm/AFLW2000/"
BIWI_PRETRAINED = "C:/Users/rossm/OneDrive - GMIT/Year 4/Final Year Project/image_processing_code/models/biwi_model_pretrained.h5"

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 25
# LEARNING_RATE = 10e-5
# EPSILON = 1e-8
ALPHA = 0.5  # Alpha is the coefficient to be applied to the regression (mse) loss

idx_tensor = [idx for idx in range(BIN_NUM)]
idx_tensor = tf.Variable(np.array(idx_tensor, dtype=np.float32))  # tensor of bins


def loss_angle(true_labels, predicted_labels, alpha=ALPHA):
    # classification loss
    # y_true_bin = true_labels[:, 0]
    # y_true_bin = tf.cast(y_true_bin, tf.int64)
    # y_true_bin = tf.one_hot(y_true_bin, 66)
    # cls_loss = tf.compat.v1.losses.softmax_cross_entropy(y_true_bin, predicted_labels)
    #
    # # regression loss
    # y_true_cont = true_labels[:, 1]
    # y_pred_cont = tf.nn.softmax(predicted_labels)
    # y_pred_cont = tf.reduce_sum(y_pred_cont * idx_tensor, 1) * 3 - 99
    # mse_loss = tf.compat.v1.losses.mean_squared_error(y_true_cont, y_pred_cont)
    #
    # total_loss = cls_loss + alpha * mse_loss
    # return total_loss

    # Classification loss
    y_true_bin = true_labels[:, 0]
    y_true_bin = tf.cast(y_true_bin, tf.int64)  # Casting elements of true binned labels to 64 bit integers
    y_true_bin = tf.one_hot(y_true_bin, BIN_NUM)  # Converting the true binned labels to onehot encoding

    # softmax_pred = tf.nn.softmax(predicted_labels)  # Carrying out softmax function on the predicted labels
    cls_loss = tf.keras.losses.categorical_crossentropy(y_true_bin, predicted_labels)

    # Regression loss
    y_true_cont = true_labels[:, 1]
    reduced_pred = tf.reduce_sum(predicted_labels * idx_tensor, 1) * 3 - 99  # reducing 1st axis of predicted labels

    mse_loss = tf.keras.losses.mean_squared_error(y_true_cont, reduced_pred)

    angle_loss = cls_loss + alpha * mse_loss

    return angle_loss


inputs = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))

# # VGG16 backbone
net = VGG16(weights=None, include_top=False)
feature = net(inputs)

# feature = Conv2D(filters=64, kernel_size=(11, 11), strides=4, padding='same', activation=tf.nn.relu)(inputs)
# feature = MaxPool2D(pool_size=(3, 3), strides=2)(feature)
# feature = Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation=tf.nn.relu)(feature)
# feature = MaxPool2D(pool_size=(3, 3), strides=2)(feature)
# feature = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
# feature = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
# feature = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)(feature)
# feature = MaxPool2D(pool_size=(3, 3), strides=2)(feature)
feature = Flatten()(feature)
feature = Dropout(0.5)(feature)
feature = Dense(units=4096, activation=tf.nn.relu)(feature)

# Output layers for pitch, roll and yaw
yaw = Dense(units=BIN_NUM, name='yaw')(feature)
pitch = Dense(units=BIN_NUM, name='pitch')(feature)
roll = Dense(units=BIN_NUM, name='roll')(feature)

model = Model(inputs=inputs, outputs=[yaw, pitch, roll])

model.compile(
    optimizer='adam',
    loss={
        'yaw': loss_angle,
        'pitch': loss_angle,
        'roll': loss_angle,
    }
)

# model.load_weights(BIWI_PRETRAINED)

aflw2000 = datasets.AFLW2000(AFLW2000_DATA_DIR, '/filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

model.summary()

model.fit(x=aflw2000.data_generator(),
          epochs=EPOCHS,
          steps_per_epoch=aflw2000.train_num // BATCH_SIZE,
          max_queue_size=10,
          workers=1,
          verbose=1)

model.save(MODELS_DIR + "hopenet2.h5")
