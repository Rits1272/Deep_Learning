#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical # For Hot Encoding our labels
import random 


# In[77]:


np.random.seed(0)


# In[78]:


(X_train, y_train), (X_test, y_test) = mnist.load_data() # 60,000 sample images with their labels


# In[79]:


assert(X_train.shape[0] == y_train.shape[0]), "The number of images are not equal to number of labels"
assert(X_test.shape[0] == y_test.shape[0]), "The number of images are not equal to number of labels"
assert(X_train.shape[1:] == (28,28)), 'The dimensions of the images are not 28 x 28'
assert(X_test.shape[1:] == (28,28)), 'The dimensions of the images are not 28 x 28'


# In[80]:


num_of_samples = []

cols = 5
num_classes = 10

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5,10)) # Creating grid.
fig.tight_layout()

for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected - 1))], cmap = plt.get_cmap('gray'))
        axs[j][i].axis('off')
        if i==2:
            axs[j][i].set_title(str(j))
            num_of_samples.append(len(x_selected))


# In[81]:


print(num_of_samples)
plt.figure(figsize=(12,4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title('Distribution of the training dataset')
plt.xlabel('class number')
plt.ylabel('No. of images')


# In[94]:


# Hot Encoding the data
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test)


# In[95]:


# Normalizing our dataset
X_train = X_train/255
X_test = X_test/255


# In[96]:


# Formatting the dataset
num_pixels = 28 * 28
X_train = X_train.reshape(X_train.shape[0], num_pixels)
X_test = X_test.reshape(X_test.shape[0], num_pixels)


# In[140]:


def create_model():
    model = Sequential()
    model.add(Dense(30, input_dim=num_pixels,activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[141]:


model = create_model()


# In[142]:


h = model.fit(x=X_train, y=y_train, verbose=1, validation_split=0.1, epochs=15, batch_size=200, shuffle=1)


# In[143]:


plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.title('Accuracy')
plt.xlabel('epochs')


# In[144]:


import requests
from PIL import Image
URL = 'https://www.researchgate.net/profile/Jose_Sempere/publication/221258631/figure/fig1/AS:305526891139075@1449854695342/Handwritten-digit-2.png'
response = requests.get(URL, stream=True)
img = Image.open(response.raw)
plt.imshow(img)


# In[145]:


import cv2
img_array = np.asarray(img)
resized = cv2.resize(img_array, (28,28))
gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
image = cv2.bitwise_not(gray_scale)
plt.imshow(image, cmap=plt.get_cmap('gray'))


# In[146]:


# Normalizing the image to be predicted
image = image/255
image = image.reshape(1, 784)


# In[147]:


prediction = model.predict_classes(image)
print('Predicted digit is : ', str(prediction))


# In[138]:





# In[ ]:




