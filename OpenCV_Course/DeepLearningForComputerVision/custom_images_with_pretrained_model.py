from keras.models import load_model
from keras_preprocessing import image
import numpy as np


model = load_model('../../Computer-Vision-with-Python/06-Deep-Learning-Computer-Vision/cat_dog_100epochs.h5')

dog_file = '../CATS_DOGS/test/DOG/11957.jpg'

img = image.load_img(dog_file, target_size=(150,150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img/255

print(model.predict_classes(img))