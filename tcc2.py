# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


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
    
def build_dataset(train_fraq=0.65, 
                  n_steps=24*30, 
                  n_horizon=24, 
                  batch_size=256, 
                  shuffle_buffer=500, 
                  expand_dims=False):

    
    tf.random.set_seed(23)
    
    path="./input/energyconsumptiongenerationpricesandweather/energy_dataset.csv"
    df_mv = pd.read_csv(path)

    data  = df_mv[['time','total load actual', 'price actual']]
    hours, day, months = make_time_features(data.time)
    data = pd.concat([data.drop(['time'], axis=1), hours, day, months], axis=1)
        
    data = clean_data(data)
    

    mm = MinMaxScaler()
    data = mm.fit_transform(data)
    
    train_data, val_data, test_data = split_data(data, train_fraq=train_fraq, test_len=8760)
    
    train_ds = window_dataset(train_data, n_steps, n_horizon, batch_size, shuffle_buffer , expand_dims=expand_dims)
    val_ds = window_dataset(val_data, n_steps, n_horizon, batch_size, shuffle_buffer, expand_dims=expand_dims)
    test_ds = window_dataset(test_data, n_steps, n_horizon, batch_size, shuffle_buffer, expand_dims=expand_dims)

    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = build_dataset()
def get_params():
    lr = 3e-4
    n_steps=24*30
    n_horizon=24
    n_features=5

    return n_steps, n_horizon, n_features, lr

model_configs = dict()

def cfg_model_run(model, history, test_ds):
    return {"model": model, "history" : history, "test_ds": test_ds}


def run_model(model_name, model_func, model_configs, epochs):
    
    n_steps, n_horizon, n_features, lr = get_params()
    train_ds, val_ds, test_ds = build_dataset(n_steps=n_steps, n_horizon=n_horizon)

    model = model_func(n_steps, n_horizon, n_features, lr=lr)

    model_hist = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model_configs[model_name] = cfg_model_run(model, model_hist, test_ds)
    return test_ds

def dnn_model(n_steps, n_horizon, n_features, lr):
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(n_steps, n_features)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name='dnn')
    
    loss=tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
    
    return model


dnn = dnn_model(*get_params())
dnn.summary()

def cnn_model(n_steps, n_horizon, n_features, lr=3e-4):
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(48, kernel_size=2, strides = 1, activation='relu', input_shape=(n_steps,n_features)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(48, activation='relu',),
        tf.keras.layers.Dense(n_horizon)
    ], name="CNN")
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
    
    return model

cnn = cnn_model(*get_params())
cnn.summary()

def lstm_model(n_steps, n_horizon, n_features, lr):
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, input_shape=(n_steps,n_features), 
         return_sequences=True),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(n_horizon)
    ], name='lstm')
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
    
    return model

lstm = lstm_model(*get_params())
lstm.summary()


def lstm_cnn_model(n_steps, n_horizon, n_features, lr):
    
    tf.keras.backend.clear_session()
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(100, kernel_size=2, strides = 1, padding = 'causal', activation='relu', input_shape=(n_steps,n_features)),
        tf.keras.layers.LSTM(100, activation='relu', return_sequences=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(n_horizon)
    ], name="lstm_cnn")
    
    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    
    model.compile(loss=loss, optimizer='adam', metrics=['mae',tf.keras.metrics.RootMeanSquaredError()])
    
    return model

lstm_cnn = lstm_cnn_model(*get_params())
lstm_cnn.summary()

model_configs=dict()
run_model("dnn", dnn_model, model_configs, epochs=150)
run_model("cnn", cnn_model, model_configs, epochs=150)
run_model("lstm", lstm_model, model_configs, epochs=150)
run_model("lstm_cnn", lstm_cnn_model, model_configs, epochs=150)

legend = list()

fig, axs = plt.subplots(1, 4, figsize=(25,5))

def plot_graphs(metric, val, ax, upper):
    ax.plot(val['history'].history[metric])
    ax.plot(val['history'].history[f'val_{metric}'])
    ax.set_title(key)
    ax.legend([metric, f"val_{metric}"])
    ax.set_xlabel('epochs')
    ax.set_ylabel(metric)
    ax.set_ylim([0, upper])
    
for (key, val), ax in zip(model_configs.items(), axs.flatten()):
    plot_graphs('loss', val, ax, 0.2)
print("Loss Curves")
plt.show()
print("MAE Curves")
fig, axs = plt.subplots(1, 4, figsize=(25,5))
for (key, val), ax in zip(model_configs.items(), axs.flatten()):
    plot_graphs('mae', val, ax, 0.6)

print("RSME Curves")
fig, axs = plt.subplots(1, 4, figsize=(25,5))
for (key, val), ax in zip(model_configs.items(), axs.flatten()):
    plot_graphs('root_mean_squared_error', val, ax, 0.6)
plt.show()

names = list()
performance = list()

for key, value in model_configs.items():
    names.append(key)
    mae = value['model'].evaluate(value['test_ds'])
    performance.append([mae[1],mae[2]])
    
performance_df = pd.DataFrame(performance, index=names, columns=['mae','rsme'])
performance_df['error_mw'] = performance_df['mae'] * df['total load forecast'].mean()
print(performance_df)    

names = list()
performance = list()

#print table for validation set
for key, value in model_configs.items():
    names.append(key)
    mae = value['model'].evaluate(val_ds)
    performance.append([mae[1],mae[2]])
    
performance_df = pd.DataFrame(performance, index=names, columns=['mae','rsme'])
performance_df['error_mw'] = performance_df['mae'] * df['total load forecast'].mean()
print(performance_df)   
