"""
First, you need to collect training data and deploy it like this.
e.g. 3-classes classification Pizza, Poodle, Rose
  ./data/
    train/
      pizza/
        pizza1.jpg
        pizza2.jpg
        ...
      poodle/
        poodle1.jpg
        poodle2.jpg
        ...
      rose/
        rose1.jpg
        rose2.jpg
        ...
    validation/
      pizza/
        pizza1.jpg
        pizza2.jpg
        ...
      poodle/
        poodle1.jpg
        poodle2.jpg
        ...
      rose/
        rose1.jpg
        rose2.jpg
        ...
"""

import sys
import os
import numpy as np
from keras.preprocessing import image
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential

from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks

# DEV = False
# argvs = sys.argv
# argc = len(argvs)

# if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
#   DEV = True

# if DEV:
#   epochs = 2
# else:
#   epochs = 20

import os
import os.path
import shutil

# train_data_path = "training"
# import csv
# images = [f for f in os.listdir(train_data_path) if os.path.isfile(os.path.join(train_data_path, f))]

# for image in images:
#     folder_name = image.filename

#     new_path = os.path.join(folder_path, folder_name)
#     if not os.path.exists(new_path):
#        os.makedirs(new_path)

# with open('solution.csv') as ourCSV:
#     lines = ourCSV.readlines()
#     myint=0
#     for line in lines[1:]:
#         #print(line)
#         something = line.split(',')
#         # print(something[1][:-1])
#         os.rename("training\\"+ str(myint+1)+'.png',"training\\"+ str(myint+1)+'.jpg')
#         old_image_path = os.path.join(train_data_path, str(myint+1)+'.jpg')
#         new_image_path = os.path.join('training_set\\'+something[1][:-1], str(myint+1)+'.jpg')
#         shutil.move(old_image_path, new_image_path)
#         if myint == 4999 :
#             break
#         myint=myint+1




"""
Parameters
"""


# lr = 0.0004
vikings = Sequential()
vikings.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 3), activation='relu'))
vikings.add(MaxPooling2D(pool_size=(2,2)))
vikings.add(Convolution2D(32, 3, 3,  activation='relu'))
vikings.add(MaxPooling2D(pool_size=(2,2)))
vikings.add(Convolution2D(64, 3, 3,  activation='relu'))
vikings.add(MaxPooling2D(pool_size=(2,2)))
vikings.add(Convolution2D(64, 3, 3,  activation='relu'))
vikings.add(MaxPooling2D(pool_size=(2,2)))
vikings.add(Convolution2D(128, 3, 3,  activation='relu'))
vikings.add(MaxPooling2D(pool_size=(2,2)))
vikings.add(Flatten())
vikings.add(Dense(output_dim=128, activation = 'relu'))
vikings.add(Dense(output_dim=6, activation = 'softmax'))

vikings.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'training_set',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical')
# test_set = test_datagen.flow_from_directory('testing_set',
#                                             target_size = (128, 128),
#                                             batch_size = 16,
#                                             class_mode = 'categorical')
0
'''validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical')'''

"""
Tensorboard log
"""
'''log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cbks = [tb_cb]'''

vikings.fit_generator(
    train_generator,
    steps_per_epoch=5000,
    epochs=1)

target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
vikings.save('./models/vikings.h5')
vikings.save_weights('./models/weights.h5')



# model_path = './models/vikings.h5'
# model_weights_path = './models/weights.h5'
# model = load_model(model_path)
# model.load_weights(model_weights_path)

# def predict(file):
#   x = load_img(file, target_size=(64,64))
#   x = img_to_array(x)
#   x = np.expand_dims(x, axis=0)
#   array = model.predict(x)
#   result = array[0]
#   answer = np.argmax(result) 
#   return answer+1



# # it=0
# # prediction=[]





# for it in range(0,10):
#     print(predict("testing\\"+ str(it+1)+'.png'))

    

