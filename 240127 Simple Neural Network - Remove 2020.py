# cd OneDrive/Desktop/NUS/"2024 Final Year Project"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

print(tf.__file__)

# parameters
metric = 'mean_squared_error'
remove_2020 = False
shuffle_before_split = False
lags = 20
separate_returns = True

loss_fn = metric
input_dim_lags = lags

daily_returns = pd.read_csv('data/daily_returns.csv', usecols = range(1, 29), parse_dates=['date'])
daily_returns_long = pd.read_csv('data/daily_returns_long.csv', usecols = range(1, 4), parse_dates=['date'])
garch_fit = pd.read_csv('data/garch_fit_results_AAPL_DIS.csv', usecols = range(1, 4), parse_dates=['date'])

# Focus on AAPL and DIS
daily_returns_AAPL_DIS = daily_returns[["date", "AAPL", "DIS"]]

# standardize returns
AAPL_predicted_vol = garch_fit.loc[garch_fit["ticker"] == "AAPL"]
DIS_predicted_vol = garch_fit.loc[garch_fit["ticker"] == "DIS"]

daily_returns_AAPL_DIS = daily_returns_AAPL_DIS.iloc[-5031:,]
daily_returns_AAPL_DIS.reset_index(drop=True, inplace=True)
AAPL_predicted_vol.reset_index(drop=True, inplace=True)
DIS_predicted_vol.reset_index(drop=True, inplace=True)

daily_returns_AAPL_DIS['AAPL'] = daily_returns_AAPL_DIS['AAPL'].div(AAPL_predicted_vol['fitted_sigma'])
daily_returns_AAPL_DIS['DIS'] = daily_returns_AAPL_DIS['DIS'].div(DIS_predicted_vol['fitted_sigma'])

# multiply returns of AAPL and DIS
daily_returns_AAPL_DIS['product_returns'] = daily_returns_AAPL_DIS['AAPL'] * daily_returns_AAPL_DIS['DIS']

# Remove 2020
if remove_2020:
    daily_returns_AAPL_DIS = daily_returns_AAPL_DIS.iloc[:4781,]

if not separate_returns:
    # make a new dataframe with columns being lags 
    product_returns_lags = pd.DataFrame()
    for i in range(0, lags + 1):
        product_returns_lags[f'lag_{i}'] = daily_returns_AAPL_DIS['product_returns'].shift(i)

    # define X and y
    X = product_returns_lags.iloc[lags:, 1:]
    y = product_returns_lags.iloc[lags:, 0]

else:
    # make a new dataframe with columns being lags 
    returns_lags = pd.DataFrame()
    for i in range(1, lags + 1):
        returns_lags[f'AAPL_lag_{i}'] = daily_returns_AAPL_DIS['AAPL'].shift(i)
        returns_lags[f'DIS_lag_{i}'] = daily_returns_AAPL_DIS['DIS'].shift(i)

    # define X and y
    X = returns_lags.iloc[lags:,]
    y = daily_returns_AAPL_DIS.iloc[lags:, 3]

    input_dim_lags = lags * 2

# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=shuffle_before_split)

print((y_test ** 2).mean())
print((abs(y_test)).mean())

# Define the model
model = Sequential()
#model.add(Dense(5, input_dim=input_dim_lags, activation='relu'))
#model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss=loss_fn, optimizer='adam', metrics=[metric, 'mean_absolute_error', r2_score])

# Fit the model
X_train, y_train = shuffle(X_train, y_train)
history = model.fit(X_train, y_train, epochs=40, batch_size=10, validation_data=(X_test, y_test))

# Evaluate the model
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Plot the training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
