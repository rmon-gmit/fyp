import cv2
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense


data = genfromtxt('../DATA/bank_note_data.txt', delimiter=',')

# the data here contains a vector of features that have been extracted from bank notes
# these will be used to figure out if a bank note is real or not
# the last column in the vector denotes whether the not was real or not (1 or 0)

# seperating out the label ( last column ) from the features
labels = data[:,4]

features = data[:,0:4]

#
X = features
y = labels

# splitting data into training and test set
# train_test_split function does this for us
# this function takes in the features and labels, then decides what percentage of the data
# should be allocated to the test sets (X_test 33% X_train 66% and y_test 33% y_train 66%),
# random_state is just the random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# scaling data forces all feature data to fall within a ceatain range
# this can help the neural net perform better

scaler_object = MinMaxScaler()
scaler_object.fit(X_train)  # only fitting it to training data so it doesnt see the test data
MinMaxScaler(copy=True, feature_range=(0, 1))
scaled_X_train = scaler_object.transform(X_train)    # will return scaled x train
scaled_X_test = scaler_object.transform(X_test)    # will return scaled x test

# building a simple network with keras
model = Sequential()
# (activation is the activation function we choose)
model.add(Dense(4, input_dim=4, activation='relu'))     # input layer
model.add(Dense(8, activation='relu'))                  # hidden layer
model.add(Dense(1, activation='sigmoid'))               # output layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# training the model
# 1 epoch means id has gone through the entire training data once
# more epochs means more accurate model
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)

predictions = model.predict_classes(scaled_X_test)

# predictions = np.argmax(model.predict(scaled_X_test), axis=-1)
print('\n///// CONFUSTION MATRIX /////')
print(confusion_matrix(y_test, predictions))
print('\n///// CLASSIFICATION REPORT /////')
print(classification_report(y_test, predictions))


