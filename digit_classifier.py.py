#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist  # 28x28 images of handwritten digits 0-9 28 rows and 28 columns

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential() # Most common model

# Whenever you need to convert a multidimensional 
# tensor into a single 1-D tensor, we can use Flatten.

# Input Layer
model.add(tf.keras.layers.Flatten())
# Hidden Layer 1
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # 128 is the # of neurons
# Hidden Layer 2
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Output Layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax ))

model.compile(optimizer='adam', 
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_accuracy = model.evaluate(x_test, y_test)
print(val_loss)
print(val_accuracy)

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()
