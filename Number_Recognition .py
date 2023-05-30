#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist


# In[4]:


# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[5]:


# Preprocess the data
X_train = X_train / 255.0
X_test = X_test / 255.0


# In[7]:


# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])


# In[8]:


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# In[10]:


# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')


# In[11]:


# Make predictions on new images
predictions = model.predict(X_test[:5])
predicted_labels = [tf.argmax(prediction).numpy() for prediction in predictions]
print(f'Predicted labels: {predicted_labels}')


# In[ ]:




