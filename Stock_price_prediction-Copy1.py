#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# In[2]:


# Load the stock price data
df = pd.read_csv("stock.csv")
df.head()


# In[4]:


# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
print(scaled_data)


# In[5]:


# Split the data into training and test sets
train_data = scaled_data[:200]
test_data = scaled_data[200:]
print(train_data)
print(test_data)


# In[10]:


# Define the input sequences and target values for training
window_size = 60  # Number of previous time steps to consider
X_train, y_train = [], []
for i in range(window_size, len(train_data)):
    X_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape)
print(y_train.shape)


# In[11]:


# Reshape the input data for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)


# In[13]:


# Build the LSTM model
from keras.models import Sequential
from keras.layers import LSTM, Dense
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))


# In[14]:


# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)


# In[16]:


df = pd.read_csv("stock.csv")
actual_stock_price = df.iloc[:,1:2].values


# In[15]:


# Prepare the test data
inputs = df['Close'][len(df) - len(test_data) - window_size:].values.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(window_size, len(inputs)):
    X_test.append(inputs[i-window_size:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[17]:


# Make predictions on the test data
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


# In[18]:


# Plotting the actual and predicted prices
plt.plot(df['Close'][200 + window_size:].values, color='blue', label='Actual Prices')
plt.plot(predicted_prices, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[ ]:




