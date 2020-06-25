import numpy as np
import matplotlib.pyplot as plt
from solar_plant import data_prep
from solar_plant import normalize

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

full_data = data_prep()
daily = full_data.resample('D').mean()

X = normalize(np.array(daily['ENERGY']))

n=len(X)
window = 7

# Split 80/20 into train/test data
last = int(n/5.0)
Xtrain = X[:-last]
Xtest = X[-last-window:]

# Store window number of points as a sequence
xin = []
next_X = []
for i in range(window,len(Xtrain)):
    xin.append(Xtrain[i-window:i])
    next_X.append(Xtrain[i])

# Reshape data to format for LSTM
xin, next_X = np.array(xin), np.array(next_X)
xin = xin.reshape(xin.shape[0], xin.shape[1], 1)



# Initialize LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(xin.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=264))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#model = Sequential()
#model.add(LSTM(units=100, return_sequences= True, input_shape=(xin.shape[1],1)))
#model.add(LSTM(units=80, return_sequences=True))
#model.add(LSTM(units=30))
#model.add(Dense(1, activation='softmax'))
#model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#model.summary()


##
history = model.fit(xin, next_X, epochs = 200, batch_size = 50,verbose=0)

plt.figure()
plt.ylabel('loss'); plt.xlabel('epoch')
plt.semilogy(history.history['loss'])

##
# Store "window" points as a sequence
xin = []
next_X1 = []
for i in range(window,len(Xtest)):
    xin.append(Xtest[i-window:i])
    next_X1.append(Xtest[i])

# Reshape data to format for LSTM
xin, next_X1 = np.array(xin), np.array(next_X1)
xin = xin.reshape((xin.shape[0], xin.shape[1], 1))

# Predict the next value (1 step ahead)
X_pred = model.predict(xin)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(X_pred,':',label='LSTM')
plt.plot(next_X1,'--',label='Actual')
plt.legend()

##

# Using predicted values to predict next step
X_pred = Xtest.copy()
for i in range(window,len(X_pred)):
    xin = X_pred[i-window:i].reshape((1, window, 1))
    X_pred[i] = model.predict(xin)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(X_pred[window:],':',label='LSTM')
plt.plot(next_X1,'--',label='Actual')
plt.legend()