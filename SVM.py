
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import math

import os
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

path="./input/energyconsumptiongenerationpricesandweather/energy_dataset.csv"
df_energy = pd.read_csv(path)

multivar_df = df_energy[['time','total load actual', 'price actual']]
multivar_df.head()

df = df_energy[["total load forecast","total load actual"]]
df_filled = df.interpolate("linear")
mm = MinMaxScaler()
df_scaled = mm.fit_transform(df_filled)

df_prep = pd.DataFrame(df_scaled, columns=df.columns)
y_true = df_prep["total load actual"]
y_pred_forecast = df_prep["total load forecast"]

def clean_data(series):
    series_filled = series.interpolate(method='linear')
    return series_filled
        
    
def min_max_scale(dataframe):
    mm = MinMaxScaler()
    return mm.fit_transform(dataframe)

def make_time_features(series):
    times = series.apply(lambda x: x.split('+')[0])
    datetimes = pd.DatetimeIndex(times)
    hours = datetimes.hour.values
    day = datetimes.dayofweek.values
    months = datetimes.month.values
    hour = pd.Series(hours, name='hours')
    dayofw = pd.Series(day, name='dayofw')
    month = pd.Series(months, name='months')
    
    return hour, dayofw, month

hour, day, month = make_time_features(multivar_df.time)

def split_data(series, train_fraq, test_len=8760):
    test_slice = len(series)-test_len
    test_data = series[test_slice:]
    train_val_data = series[:test_slice]
    train_size = int(len(train_val_data) * train_fraq)
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]
    
    return train_data, val_data, test_data


multivar_df = clean_data(multivar_df)

hours, day, months = make_time_features(multivar_df.time)
multivar_df = pd.concat([multivar_df.drop(['time'], axis=1), hours, day, months], axis=1)

multivar_df = min_max_scale(multivar_df)
train_multi, val_multi, test_multi = split_data(multivar_df, train_fraq=0.65, test_len=8760)

def window_dataset(data, n_steps, n_horizon, batch_size, shuffle_buffer, expand_dims=False):

    window = n_steps + n_horizon
    if expand_dims:
        ds = tf.expand_dims(data, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(ds)
    else:
        ds = tf.data.Dataset.from_tensor_slices(data)
    
    ds = ds.window(window, shift=n_horizon, drop_remainder=True)
    
    ds = ds.flat_map(lambda x : x.batch(window))
    ds = ds.shuffle(shuffle_buffer)    
    

    ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:, :1]))

    ds = ds.batch(batch_size).prefetch(1)
    
    return ds

tf.random.set_seed(42)

n_steps = 72
n_horizon = 24
batch_size = 1
shuffle_buffer = 100


ds = window_dataset(train_multi, n_steps, n_horizon, batch_size, shuffle_buffer)

path="./input/energyconsumptiongenerationpricesandweather/energy_dataset.csv" 
df_mv = pd.read_csv(path) 
 
data  = df_mv[['time','total load actual', 'price actual']] 
hours, day, months = make_time_features(data.time) 
data = pd.concat([data.drop(['time'], axis=1), hours, day, months], axis=1) 
data = clean_data(data) 
 
 
test_slice = len(data)-8760 
 
testData = data[test_slice:] 
trainData = data[:test_slice] 
XTrainData = trainData[['hours','dayofw','months', 'price actual']] 
YTrainData =  trainData[['total load actual']] 
 
XTestData = testData[['hours','dayofw','months', 'price actual']] 
YTestData =  testData[['total load actual']] 
 
regr = svm.SVR(kernel='poly') 
regr.fit(XTrainData, YTrainData['total load actual']) 
resultSVM = regr.predict(XTestData) 
err = []

mse = mean_squared_error(resultSVM, list(YTestData['total load actual']))
mae = mean_absolute_error(resultSVM, list(YTestData['total load actual']))
err = 0 
for i in range(0,len(resultSVM)): 
    err = err + abs(resultSVM[i] - list(YTestData['total load actual'])[i]) 
 
print(err) 
print(mae)
print(mse)   
rmse = math.sqrt(mse)
print(rmse)
print(rmse)