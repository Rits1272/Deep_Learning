#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential # Sequential is a linear type of layer and is most common
from keras.layers import Dense # In dense layer every node or neuron is connected to every other neuron.
from keras.optimizers import Adam # Adam is basically a combination of two extensions of SGD : Adagrad and RMSprop
# Adam is much more efficient than batch size gradient descent also known as Vanilla Gradient Descent.
# Adam computes a adaptive learning rate so that process becomes fast and efficient.
# In case of a vanilla gradient descent we specify a constant gradient descent.

n_pts = 500
np.random.seed(0)

Xa = np.array([np.random.normal(13,2,n_pts), np.random.normal(12,2,n_pts)]).T # Top region
Xb = np.array([np.random.normal(8,2,n_pts), np.random.normal(6,2,n_pts)]).T # Bottom region

X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T # labels =  0, 1 - (top,bottom)

model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))
adam = Adam(lr=0.1)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# metrics are simply used to see the model in each epoch
h = model.fit(x=X, y=y, verbose=1, batch_size=50, epochs=500, shuffle='true') # Verbose will display a progress bar for each epoch
# shuffle='true' helps us in avoiding the local minima of the data

# Analyzing the model

plt.plot(h.history['acc'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'])

plt.plot(h.history['loss'])
plt.title('loss')
plt.xlabel('epoch')
plt.legend(['loss'])

# Displaying the model

def plot_decision_boundary(X, y,model):
    x_span = np.linspace(min(X[:,0] - 1), max(X[:,0]) + 1, 50)
    y_span = np.linspace(min(X[:,1] - 1), max(X[:,1]) + 1, 50)
    xx, yy = np.meshgrid(x_span, y_span) # meshhgrid returns a 2D array of the repeated row or column.
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
    
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])

x = 7.5
y = 0
point = np.array([[x, y]])
prediction = model.predict(point)
print(prediction)

if prediction < 0.5:
    color = 'b'
else:
    color='r'
    
plt.plot(x, y, marker='o', markersize=10, color=color)
plt.show()