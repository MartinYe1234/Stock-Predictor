import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from google.colab import drive

drive.mount('/content/drive')

"""helper functions"""
#function to help visualise results
def plot_data(original, predictions):
  #data must be reshaped and untransformed in order to be graphed
  original = np.reshape(original, (original.shape[0], 1))
  original = scaler.inverse_transform(original)
  predictions = np.reshape(predictions, (predictions.shape[0],1))
  predictions = scaler.inverse_transform(predictions)
  
  plt.figure()
  plt.rcParams['figure.figsize']= (12,12)
  plt.plot(original, color = 'blue', label = 'actual price')
  plt.plot(predictions, color = 'green', label = 'predicted')
  plt.legend()
  plt.show()

#predicts the stock prices x days into the future - accuracy is questionable
def predict_ahead(x):
  #make predictions using test data
  predictions = model.predict(x_test)
  last = x_test[-1]
  for i in range(x):
    n = np.reshape(last,(1,batch_size,1))
    prediction = model.predict(n)
    #update last to include newest prediction and remove first element
    last = np.append(last,prediction)[1:]
    predictions = np.append(predictions, prediction)
  plot_data(y_test,predictions)


"""accessing and formating data"""

df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/Datasets/FB.csv")
#dates will be used for graphing
dates = df['Date'].values
test_dates = dates[int(dates.shape[0]*0.8):]

dataset = df['Open'].values
dataset = np.reshape(dataset, (-1,1))
#scale down all values to between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

#split into training and testing datasets - 80 20 split
training_dataset = dataset[:int(dataset.shape[0]*.8)]
test_dataset = dataset[int(dataset.shape[0]*.8):]

#create a data structure with some amount to timesteps and 1 output
def create_data(dataset,size):
  x_data, y_data = [], []
  for i in range(size, dataset.shape[0]):
    x_data.append(dataset[i-size:i, 0])
    y_data.append(dataset[i, 0])
  return np.array(x_data), np.array(y_data)

batch_size = 60
# Creating a data structure with 60 timesteps and 1 output
x_train, y_train = create_data(training_dataset, batch_size)
x_test, y_test = create_data(test_dataset, batch_size)
#reshape so that the LSTM layer can use the data
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

"""creating the model"""
model = Sequential()

model.add(LSTM(units = 400, input_shape=(x_train.shape[1],1), return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 400, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 400, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 400))
model.add(Dropout(0.3))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x_train, y_train, epochs = 25, batch_size = 32)
#make prediction
predict_ahead(3)
