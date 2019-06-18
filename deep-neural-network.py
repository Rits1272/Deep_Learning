#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[31]:


np.random.seed(0)


# In[32]:


n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

plt.scatter(X[y==0, 0], X[y == 0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])


# In[33]:


model = Sequential()
model.add(Dense(4, input_shape=(2, ), activation='sigmoid')) # First Hidden Layer
model.add(Dense(1, activation='sigmoid')) # Output layer
model.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])


# In[34]:


h = model.fit(x=X, y=y, verbose=1, batch_size=20, epochs=100, shuffle='true')


# In[35]:


plt.plot(h.history['acc'])
plt.xlabel('epoch')
plt.legend(['accuracy'])
plt.title('accuracy')


# In[37]:


plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend(['loss'])
plt.title('loss')


# In[48]:


# Displaying the decision boundary
def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:,0]) + 0.25)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:,1]) + 0.25)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


# In[54]:


plot_decision_boundary(X, y, model)
# Prediction
x = 0.1
y = 0
point = np.array([[x,y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=15, color='r')
print(prediction)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])


# In[ ]:





# In[ ]:




