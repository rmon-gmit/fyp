import math

import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50


from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from keras.models import Model, Input
from keras.models import load_model
from tensorflow.python.keras.callbacks import LearningRateScheduler

LEARNING_RATE = 10e-5
EPSILON = 1e-8
ALPHA = 0.5     # Alpha is the coefficient to be applied to the regression (mse) loss

class HopeNet:

    def __init__(self, dataset, input_size=0, num_bins=0, batch_size=0, model_path="", new=True):
        self.dataset = dataset
        self.input_size = input_size
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.idx_tensor = [idx for idx in range(self.num_bins)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32)) # tensor of bins
        if new:
            self.model = self.__create_model()
        else:
            self.model = self.__load_model(model_path)

    def __create_model(self):

        inputs = Input(shape=(self.input_size, self.input_size, 3))

        # VGG16 backbone
        net = VGG16(weights=None, include_top=False)
        feature = net(inputs)


        # AlexNet CNN backbone
        # feature = Conv2D(filters=64, kernel_size=(11, 11), strides=4, padding='same', activation='relu')(inputs)
        # feature = MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        # feature = Conv2D(filters=192, kernel_size=(5, 5), padding='same', activation='relu')(feature)
        # feature = MaxPool2D(pool_size=(3, 3), strides=2)(feature)
        # feature = Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(feature)
        # feature = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(feature)
        # feature = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu')(feature)
        # feature = MaxPool2D(pool_size=(3, 3), strides=2)(feature)

        feature = Flatten()(feature)
        feature = Dropout(0.5)(feature)

        # feature = Dense(units=4096, activation=tf.nn.relu)(feature)

        # Output layers for pitch and yaw
        pitch = Dense(units=self.num_bins, name='pitch', activation='softmax')(feature)
        yaw = Dense(units=self.num_bins, name='yaw', activation='softmax')(feature)

        model = Model(inputs=inputs, outputs=[yaw, pitch])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON),
            loss={
                'pitch': self.__loss_angle,
                'yaw': self.__loss_angle
            }
        )

        return model

    def __load_model(self, model_path):
        model = load_model(model_path, compile=False)
        return model

    def __loss_angle(self, true_labels, predicted_labels, alpha=ALPHA):
        """ Calculate the multi-part loss: classification_loss + alpha * regression_loss
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
        # softmax_pred = tf.nn.softmax(predicted_labels)  # Carrying out softmax function on the predicted labels

        cls_loss = tf.keras.losses.categorical_crossentropy(y_true_bin, predicted_labels)

        # Cegression loss
        y_true_cont = true_labels[:, 1]
        reduced_pred = tf.reduce_sum(predicted_labels * self.idx_tensor, 1) * 3 - 99    # reducing 1st axis of predicted labels

        mse_loss = tf.keras.losses.mean_squared_error(y_true_cont, reduced_pred)

        angle_loss = cls_loss + alpha * mse_loss

        return angle_loss

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

    def test(self, face_imgs):
        batch_x = np.array(object=face_imgs, dtype=np.float32)

        predictions = self.model.predict(x=batch_x)
        predictions = np.asarray(predictions)   # predictions is a tensor of values corresponding to how likely the image fits into one of 66 bins for pitch and yaw

        pred_bin_yaw = predictions[0, :, :]
        pred_bin_pitch = predictions[1, :, :]

        softmax_yaw = tf.nn.softmax(pred_bin_yaw)  # applying a softmax activation to the yaw predictions
        softmax_pitch = tf.nn.softmax(pred_bin_pitch)    # applying a softmax activation to the pitch predictions

        red_yaw = tf.reduce_sum(softmax_yaw * self.idx_tensor, 1)   # reducing the 1st axis of yaw
        red_pitch = tf.reduce_sum(softmax_pitch * self.idx_tensor, 1)   # reducing the 1st axis of pitch

        pred_cont_yaw = red_yaw * 3 - 99   # calculating continuous yaw predictions
        pred_cont_pitch = red_pitch * 3 - 99   # calculating continuous pitch predictions

        print("Pitch: " + str(pred_cont_pitch[0]) + "Yaw: " + str(pred_cont_yaw[0]))

        return pred_cont_yaw[0], pred_cont_pitch[0]
