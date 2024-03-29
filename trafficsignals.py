# -*- coding: utf-8 -*-
"""TrafficSignals.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ticc9oitRuPQYUDHTcWy01Xf2PQtupHf
"""

!git clone https://bitbucket.org/jadslim/german-traffic-signs

!ls german-traffic-signs

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random

np.random.seed(0)

with open('german-traffic-signs/train.p','rb') as f:
  train_data = pickle.load(f)
  
with open('german-traffic-signs/test.p','rb') as f:
  test_data = pickle.load(f)
  
with open('german-traffic-signs/valid.p','rb') as f:
  val_data = pickle.load(f)

X_train, y_train = train_data['features'], train_data['labels']
X_test, y_test = test_data['features'], test_data['labels']
X_val, y_val = val_data['features'], val_data['labels']

assert(X_train.shape[0] == y_train.shape[0]),'The number of images are not equal to no. of lables'
assert(X_test.shape[0] == y_test.shape[0]),'The number of images are not equal to no. of lables'
assert(X_val.shape[0] == y_val.shape[0]),'The number of images are not equal to no. of lables'
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images is not 32*32*3"
assert(X_test.shape[1:] == (32,32,3)), "The dimensions of the images is not 32*32*3"
assert(X_val.shape[1:] == (32,32,3)), "The dimensions of the images is not 32*32*3"

data = pd.read_csv('german-traffic-signs/signnames.csv')

num_of_samples = []

cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,50))
fig.tight_layout()

for i in range(cols):
  for j,row in data.iterrows():  #(index, series)
    x_selected = X_train[y_train == j]
    axs[j][i].imshow(x_selected[random.randint(0, (len(x_selected)-1))], cmap=plt.get_cmap('gray'))
    axs[j][i].axis('off')
    if i == 2:
      axs[j][i].set_title(str(j) + '-' + row['SignName'])
      num_of_samples.append(len(x_selected))

# Preprocessing data because test data is not balance.Min Images of a class is 180 and max is2010.Therefore our bata will be bias.
import cv2
plt.imshow(X_train[500])
plt.axis("off")

def grayscale(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  

img = grayscale(X_train[500])
plt.imshow(img)
plt.axis('off')
print(img.shape)

def equalize(img):
  '''
    equalizeHist function takes only gray scale image
  '''
  return cv2.equalizeHist(img)
  
img = equalize(img)
plt.imshow(img)
plt.axis('off')

def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  img = img / 255
  return img

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

plt.imshow(X_train[random.randint(0, len(X_train) - 1)])
plt.axis('off')
print(X_train.shape)

X_train = X_train.reshape(34799, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)

# for balancing images
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             zoom_range=0.2,
                             shear_range=0.1, 
                             rotation_range=10)
datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1,15,figsize=(20,5
                                   ))
fig.tight_layout()

for i in range(15):
  axs[i].imshow(X_batch[i].reshape(32,32))
  axs[i].axis('off')

def leNet_model():
  model = Sequential()
  model.add(Conv2D(60, (5,5),input_shape=(32,32,1), activation='relu'))
  model.add(Conv2D(60, (5,5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  
  model.add(Conv2D(30, (3,3), activation='relu'))
  model.add(Conv2D(30, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  #model.add(Dropout(0.5))

  
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
  return model

model = leNet_model()
print(model.summary())

h = model.fit_generator(datagen.flow(X_train, y_train, batch_size=50), steps_per_epoch=2000, epochs=10, validation_data=[X_val, y_val], shuffle=1)

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.show()

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy', score[1] * 100, '%')

'''
https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg

https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg

https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg

https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg

https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg
'''

import requests
from PIL import Image
url = 'https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg'
r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap('gray'))

#Preprocess image
 
img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap = plt.get_cmap('gray'))

print(img.shape)
 
#Reshape reshape
 
img = img.reshape(1, 32, 32, 1)
 
#Test image
print("predicted sign: "+ str(model.predict_classes(img)))

