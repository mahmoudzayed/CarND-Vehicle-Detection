import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from keras.regularizers import l2
from keras.models import Model, load_model, Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D



def read_training_data():
  inputImages = glob('ericlavigne-lanes-vehicles-data/images/*_x.png')
  outputImages = glob('ericlavigne-lanes-vehicles-data/SegImages/*_cars.png')
  inputImages.sort()
  outputImages.sort()
  assert (len(inputImages) == len(outputImages)), "x and cars files don't match"
  X_train = []
  y_train = []
  imageSize = (1280,720)
  for name in tqdm(inputImages):
    # importing the training images 
    image = cv2.cvtColor(cv2.resize(cv2.imread(name),imageSize), cv2.COLOR_BGR2RGB)
    #image = (image/255 - 0.5).astype(np.float32)
    X_train.append(image[380:600,620:])
  for name in tqdm(outputImages):
    # importing the training images 
    image = np.expand_dims(cv2.cvtColor(cv2.resize(cv2.imread(name),imageSize), cv2.COLOR_BGR2GRAY), axis=2)
    #image = (image/255 - 0.5).astype(np.float32)
    y_train.append(image[380:600,620:])
  # convert the array to be numpy array
  X_train = np.array(X_train)
  y_train = np.array(y_train)
  return X_train, y_train

def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    total_memory = 4*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = round(total_memory / (1024 ** 3), 3)
    mbytes = round(total_memory / (1024 ** 2), 3)
    
    print('trainable_count', trainable_count, 'non_trainable_count', non_trainable_count, 'gbytes', gbytes, 'mbytes', mbytes)

def create_model():
    """Create neural network model, defining layer architecture."""
    model = Sequential()
    model.add(Conv2D(20, 5, 5, border_mode='same', input_shape=(220, 660, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(30, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(30, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(30, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(20, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(10, 5, 5, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, 5, 5, border_mode='same', W_regularizer=l2(0.01), activation='relu'))
    model.compile(loss= 'mean_squared_error', optimizer= 'adam')
    return model
 
batch_size = 7
#read images and Segented images
X_train, y_train = read_training_data()
model = create_model()
get_model_memory_usage(batch_size,model)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1500, shuffle=True)
model.save('Car_Model.h5')