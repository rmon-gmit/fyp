from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import preprocess_input
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Flatten, Dropout
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def loss_angle(self, y_true, y_pred):
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

    total_loss = cls_loss + self.alpha * mse_loss
    return total_loss


inputs = Input(shape=(224, 224, 3))  # Input Layer
input_shape = inputs.shape
print(input_shape)
# weights=None: Random initialization of weights
# include_top=None: Not including the fully-connected layer at the top of the network
# input_tensor=input_tensor: Specifying the shape of image used as input for the model
resnet = ResNet50(weights=None, include_top=False)

feature = resnet(inputs)
# flattening model from 2D to 1D
feature = Flatten()(feature)
# The dropout layer below randomly turns a percentage of neurons off during training, to combat overfitting
feature = Dropout(0.5)(feature)

yaw = Dense(200, name='yaw', activation=None)(feature)
pitch = Dense(200, name='pitch', activation=None)(feature)
roll = Dense(200, name='roll', activation=None)(feature)

model = Model(inputs=inputs, outputs=[yaw, pitch, roll])

model.compile(
    optimizer='adam',
    loss={
        'yaw': loss_angle,
        'pitch': loss_angle,
        'roll': loss_angle,
    }
)

model.summary()

batch_size = 16

image_gen = ImageDataGenerator(rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=(1 / 255),
                               horizontal_flip=True)

# NEED TO HAVE DIRECTORY OF TRAINING IMAGES
train_img_gen = image_gen.flow_from_directory(directory='C:/Users/rossm/kinect_head_pose_db/hpdb',
                                               target_size=input_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='categorical')

result = model.fit(x=datasets.data_generator(test=False),
                    epochs=1,
                    steps_per_epoch=150,
                    max_queue_size=10,
                    workers=1,
                    verbose=1)

model.save("C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\image_processing_code\\model")