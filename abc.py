
import sys
import os
import numpy as np
from keras.preprocessing import image
from PIL import Image
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks





model_path = './models/vikings.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(128,128))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result) 
  return answer+1



# it=0
# prediction=[]




with open('sample.csv', mode = 'w') as xyz:
    for it in range(0,40000):
        xyz.write(str(it+1) +','+ str(predict("testing\\"+ str(it+1)+'.png'))+'\n')
