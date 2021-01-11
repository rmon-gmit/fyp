import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, Input
from keras.models import load_model


class HopeNet:

    def __init__(self, dataset="", input_size=0, num_bins=0, batch_size=0, model_path="", new=True):
        self.dataset = dataset
        self.input_size = input_size
        self.num_bins = num_bins
        self.batch_size = batch_size
        self.idx_tensor = [idx for idx in range(self.num_bins)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        if new:
            self.model = self.create_model()
        else:
            self.model = self.load_model(model_path)

    def create_model(self):
        inputs = Input(shape=(self.input_size, self.input_size, 3))  # Input Layer

        net = VGG16(weights=None, include_top=False)

        feature = net(inputs)
        feature = Flatten()(feature)
        feature = Dropout(0.5)(feature)

        yaw = Dense(self.num_bins, name='yaw', activation=None)(feature)
        pitch = Dense(self.num_bins, name='pitch', activation=None)(feature)
        roll = Dense(self.num_bins, name='roll', activation=None)(feature)

        model = Model(inputs=inputs, outputs=[yaw, pitch, roll])

        model.compile(
            optimizer='adam',
            loss={
                'yaw': self.loss_angle,
                'pitch': self.loss_angle,
                'roll': self.loss_angle,
            }
        )

        return model

    def load_model(self, model_path):
        model = load_model(model_path, compile=False)
        return model

    def loss_angle(self, y_true, y_pred, alpha=0.5):
        """ Calculate the multi-part loss: classification_loss + alpha * regression_loss
        Args:
          y_true: the true label
          y_pred: the predicted label
          alpha: the alpha value
        Returns:
          total_loss: the multipart loss
        """

        # classification loss
        y_true_bin = y_true[:, 0]
        y_true_bin = tf.cast(y_true_bin, tf.int64)
        y_true_bin = tf.one_hot(y_true_bin, 66)
        cls_loss = tf.compat.v1.losses.softmax_cross_entropy(y_true_bin, y_pred)

        # regression loss
        y_true_cont = y_true[:, 1]
        y_pred_cont = tf.nn.softmax(y_pred)
        y_pred_cont = tf.reduce_sum(y_pred_cont * self.idx_tensor, 1) * 3 - 99
        mse_loss = tf.compat.v1.losses.mean_squared_error(y_true_cont, y_pred_cont)

        total_loss = cls_loss + alpha * mse_loss

        return total_loss

    def train(self, model_path, max_epochs=25, load_weight=True):
        self.model.summary()

        if load_weight:
            self.model.load_weights(model_path)
        else:
            self.model.fit_generator(generator=self.dataset.data_generator(test=False),
                                     epochs=max_epochs,
                                     steps_per_epoch=200,   # self.dataset.train_num // self.batch_size,
                                     max_queue_size=10,
                                     workers=1,
                                     verbose=1)

            self.model.save(model_path)

    def test(self, face_imgs):
        batch_x = np.array(face_imgs, dtype=np.float32)
        predictions = self.model.predict(batch_x)
        predictions = np.asarray(predictions)
        pred_cont_yaw = tf.reduce_sum(tf.nn.softmax(predictions[0, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_pitch = tf.reduce_sum(tf.nn.softmax(predictions[1, :, :]) * self.idx_tensor, 1) * 3 - 99
        pred_cont_roll = tf.reduce_sum(tf.nn.softmax(predictions[2, :, :]) * self.idx_tensor, 1) * 3 - 99
        print("Pitch: " + str(pred_cont_pitch) + "Yaw: " + str(pred_cont_yaw) + "Roll: " + str(pred_cont_roll))
        return pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]
