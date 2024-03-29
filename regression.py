# -*- coding: utf-8 -*-
"""Regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z_cU5VYTa3MKeHRX3zfI456biAl2st3P
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed(0)
points = 500
X = np.linspace(-3, 3, points)
y = np.sin(X) + np.random.uniform(-0.5,0.5, points)
plt.scatter(X, y)
plt.show()

model = Sequential()
model.add(Dense(50, input_dim=1, activation='sigmoid'))
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(1))

model.compile(Adam(lr=0.01), loss='mse') # Mean Squared Error
model.fit(X, y, epochs=50)

predictions = model.predict(X)
plt.plot(X, predictions, 'ro')
plt.scatter(X, y)
plt.show()

