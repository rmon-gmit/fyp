from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

single_image = x_train[0]
plt.imshow(single_image, cmap='gray_r')
plt.show()

# converting to one hot encoding
y_cat_test = to_categorical(y_test, 10)     # 10 classes (0,1,2,3,4,5,6,7,8,9)
y_cat_train = to_categorical(y_train, 10)
print(y_cat_test)
print(y_cat_train)

# the result will be an array of 0's with the fifth index a value of 1
# if we chose a 4 the array would be all zeroes except the 4th index
print(y_cat_train[0])

# normalizing the values to between 0 and 1
# this is manual way, sklearn also has the .fit method that does this for you
x_train = x_train/x_train.max()
x_test = x_test/x_test.max()

# reshaping the data to clarify there is only one colour channel
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# building and training model
model = Sequential()

# Convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28,28,1), activation='relu'))
# pooling layer
model.add(MaxPool2D(pool_size=(2,2)))
# flattening model from 2D to 1D
model.add(Flatten())
# dense layer
model.add(Dense(128, activation='relu'))
# output layer
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_cat_train, epochs=2)

model.evaluate(x_test, y_cat_test)
predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))