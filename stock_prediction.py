# -*- coding: utf-8 -*-
"""Generate your own API key. For more information check out the video on my youtube channel - 
    https://youtu.be/1JrrYqq_S0k 
    api key to be replaced in line 21,132,154
"""

# Description : this program prediscts the stock market using LSTM which is an artificial neural network

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('fivethirtyeight')

#get the stock quote
df = web.DataReader("AAPL", "av-daily", start=datetime(2010, 1, 1),end=datetime(2021, 11, 19),api_key='<replace this with api key>')
#show the data
df

# get the number of rows and columns
df.shape

#visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price history')
plt.plot(df['close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price in USD', fontsize=18)
plt.show()

# Create a new dataframe with only the close column
data = df.filter(['close'])
#convert the dataframe to numpy array
dataset = data.values
#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)

training_data_len

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

#Create the training dataset
#create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 61:
    print(x_train)
    print(y_train)
    print()

#Convert the x_train and y_train data set to numpy arrays
x_train, y_train = np.array(x_train),np.array(y_train)

#reshape the data
print(x_train)
print("x_train.shape[0]",x_train.shape[0])
print("x_train.shape[1]",x_train.shape[1])
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape

#build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model

model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create the testing dataset
#create a new array containing scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len - 60: , :]
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i,0])

#convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#getting the root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

#Plot data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('date', fontsize=18)
plt.ylabel('Closed Price USD',fontsize=18)
plt.plot(train['close'])
plt.plot(valid[['close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()

#show the valid prediction prices
valid

#Get the quote
apple_quote = web.DataReader("AAPL", "av-daily", start=datetime(2021,8,15),end=datetime(2021, 11, 22),api_key='<replace-api-key>')
#create a new data frame
new_df = apple_quote.filter(['close'])
#get the last 60 day closing price value and convert the df to array
last_60_days = new_df[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list
X_test=[]
#append the past 60 days data 
X_test.append(last_60_days_scaled)
#convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

#Get the quote
apple_quote2 = web.DataReader("AAPL", "av-daily", start=datetime(2021,11, 17),end=datetime(2021, 11, 23),api_key='<replace-api-key>')
print(apple_quote2['close'])