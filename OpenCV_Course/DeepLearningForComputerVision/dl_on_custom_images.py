import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPool2D, Dense
from keras_preprocessing import image
# from keras.models import load_model

cat4 = cv2.imread('../CATS_DOGS/train/CAT/4.jpg')
cat4 = cv2.cvtColor(cat4, cv2.COLOR_BGR2RGB)
plt.imshow(cat4)
plt.show()

dog = cv2.imread('../CATS_DOGS/train/DOG/2.jpg')
dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)
plt.imshow(dog)
plt.show()

dog_file = '../CATS_DOGS/test/DOG/11957.jpg'
img = image.load_img(dog_file, target_size=(150,150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255

image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=(1 / 255),
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')

# plt.imshow(image_gen.random_transform(dog))
# plt.show()

image_gen.flow_from_directory('../CATS_DOGS/train')

input_shape = (150, 150, 3)

# creating model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

# Overfitting happens when a model learns the detail and noise in the training data to the
# extent that it negatively impacts the performance of the model on new data. This means that
# the noise or random fluctuations in the training data is picked up and learned as concepts by
# the model. The problem is that these concepts do not apply to new data and negatively impact
# the models ability to generalize.
# The dropout layer below randomly turns a percentage of neurons off during training
# This combats overfitting issues
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

batch_size = 16

train_img_gen = image_gen.flow_from_directory('../CATS_DOGS/train',
                                               target_size=input_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

test_img_gen = image_gen.flow_from_directory('../CATS_DOGS/test',
                                             target_size=input_shape[:2],
                                             batch_size=batch_size,
                                             class_mode='binary')
# print(train_img_gen.class_indices)

results = model.fit_generator(train_img_gen, epochs=5, steps_per_epoch=150,
                              validation_data=test_img_gen, validation_steps=12)

# print(results.history['acc'])

print(model.predict_classes(img))