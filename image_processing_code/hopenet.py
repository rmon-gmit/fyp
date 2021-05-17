"""
    Name: Ross Monaghan
    File: hopenet.py
    Description: File containing head pose estimation model class and methods
    Date: 15/05/21

    ** THE FOLLOWING CODE CONTAINS SECTIONS FROM THE TENSORFLOW ADAPTATION OF THE HOPENET HEAD POSE ESTIMATION MODEL **
    ** URL: https://github.com/Oreobird/tf-keras-deep-head-pose **

    @InProceedings{Ruiz_2018_CVPR_Workshops,
    author = {Ruiz, Nataniel and Chong, Eunji and Rehg, James M.},
    title = {Fine-Grained Head Pose Estimation Without Keypoints},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2018}
    }

"""

import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.applications.vgg16 import VGG16

from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.models import Model, Input, load_model

LEARNING_RATE = 10e-5
EPSILON = 1e-8
ALPHA = 0.5  # Alpha is the coefficient to be applied to the regression (mse) loss


# Class to represent the Hopenet model
class HopeNet:

    # Constructor
    def __init__(self, dataset="", input_size=0, num_bins=0, batch_size=0, model_path="", new=True):
        self.dataset = dataset
        self.input_size = input_size
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.idx_tensor = [idx for idx in range(self.num_bins)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))  # tensor of bins
        if new:
            self.model = self.__create_model()
        else:
            self.model = self.__load_model(model_path)

    # Method to create a Hopenet model
    def __create_model(self):

        inputs = Input(shape=(self.input_size, self.input_size, 3))

        # VGG16 backbone
        net = VGG16(weights=None, include_top=False)
        feature = net(inputs)

        feature = Flatten()(feature)
        feature = Dropout(0.5)(feature)

        # feature = Dense(units=4096, activation=tf.nn.relu)(feature)

        # Output layers for pitch and yaw
        pitch = Dense(units=self.num_bins, name='pitch')(feature)
        yaw = Dense(units=self.num_bins, name='yaw')(feature)
        roll = Dense(units=self.num_bins, name='roll')(feature)

        model = Model(inputs=inputs, outputs=[yaw, pitch, roll])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON),
            loss={
                'pitch': self.__loss_angle,
                'yaw': self.__loss_angle,
                'roll': self.__loss_angle
            }
        )

        return model

    # Method to load a Hopenet model
    def __load_model(self, model_path):
        model = load_model(model_path, compile=False)
        return model

    # Loss function for use in training Hopenet on classification and regression labels
    def __loss_angle(self, true_labels, predicted_labels, alpha=ALPHA):
        """ Multi-part loss: classification_loss + alpha * regression_loss
        Args:
          true_labels: the actual binary and continuous labels
          predicted_labels: the predicted binary and continuous labels
          alpha: the coefficient for the mse result
        Returns:
          angle_loss: total loss for this specific angle (yaw or pitch)
        """

        # Classification loss
        y_true_bin = true_labels[:, 0]
        y_true_bin = tf.cast(y_true_bin, tf.int64)  # Casting elements of true binned labels to 64 bit integers
        y_true_bin = tf.one_hot(y_true_bin, self.num_bins)  # Converting the true binned labels to onehot encoding
        softmax_pred = tf.nn.softmax(predicted_labels)  # Carrying out softmax function on the predicted labels
        cls_loss = tf.keras.losses.categorical_crossentropy(y_true_bin, softmax_pred)  # classification loss

        # Regression loss
        y_true_cont = true_labels[:, 1]
        reduced_pred = tf.reduce_sum(predicted_labels * self.idx_tensor, 1) * 3 - 99  # reducing 1st axis of predicted labels

        mse_loss = tf.keras.losses.mean_squared_error(y_true_cont, reduced_pred)  # regression loss

        angle_loss = cls_loss + alpha * mse_loss  # total loss

        return angle_loss

    # Method to train a Hopenet model
    def train(self, model_path, epochs=25, load_weight=False):
        self.model.summary()

        if load_weight:
            self.model.load_weights(model_path)
        else:
            self.model.fit(x=self.dataset.data_generator(),
                           epochs=epochs,
                           steps_per_epoch=self.dataset.train_num // self.batch_size,
                           max_queue_size=10,
                           workers=1,
                           verbose=1)

            self.model.save(model_path)

    # Method to test a Hopenet model
    def test(self, face_imgs):
        batch_x = np.array(object=face_imgs, dtype=np.float32)

        pred = self.model.predict(x=batch_x)
        pred = np.asarray(pred)  # tensor of values corresponding to how likely the image fits into euler angle bins

        pred_bin_yaw = pred[0, :, :]
        pred_bin_pitch = pred[1, :, :]

        softmax_yaw = tf.nn.softmax(pred_bin_yaw)  # applying a softmax activation to the yaw pred
        softmax_pitch = tf.nn.softmax(pred_bin_pitch)  # applying a softmax activation to the pitch pred

        red_yaw = tf.reduce_sum(softmax_yaw * self.idx_tensor, 1)  # reducing the 1st axis of yaw
        red_pitch = tf.reduce_sum(softmax_pitch * self.idx_tensor, 1)  # reducing the 1st axis of pitch

        pred_cont_yaw = red_yaw * 3 - 99  # calculating continuous yaw pred
        pred_cont_pitch = red_pitch * 3 - 99  # calculating continuous pitch pred

        print("Pitch: " + str(pred_cont_pitch[0]) + "Yaw: " + str(pred_cont_yaw[0]))

        return pred_cont_yaw[0], pred_cont_pitch[0]
