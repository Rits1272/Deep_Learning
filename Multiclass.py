#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.utils.np_utils import to_categorical # For converting labels to hot-encoded form


# In[10]:


n_pts = 500
# labels = 0        1        2
centers = [[-1,1], [-1,-1], [1,-1]]
X, y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)


# In[11]:





# In[12]:


y_cat = to_categorical(y, 3) # Hot encoded labels


# In[17]:


model = Sequential()
model.add(Dense(units=3, input_shape=(2, ), activation='softmax'))
model.compile(Adam(0.1), loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(x=X, y=y_cat, batch_size=50, shuffle='true', verbose=1, epochs=10)


# In[36]:


def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
    y_span = np.linspace(min(X[:, 1]) - 1, max(y[:, 1]) + 2)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict_classes(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


# In[48]:


plot_decision_boundary(X, y_cat, model)
plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])
plt.scatter(X[y==2, 0], X[y==2, 1])
x=0.1
y=0.5
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='*', markersize=15, color='r')
plt.show()
print('Probabilities are : ', prediction)
print("Prediction is : ", np.argmax(prediction))


# In[ ]:





# In[ ]:




