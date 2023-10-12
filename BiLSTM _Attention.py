#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load data
dataset = pd.read_csv('D:\\model and data\\AL945_M.csv',header=0, 
                      low_memory=False, infer_datetime_format=True, engine='c',
                      parse_dates={'datetime':[0]},index_col=['datetime'])

dataset.shape
dataset.head(10)

d=np.array(dataset.iloc[:,0])
dmin=d.min()
dmax=d.max()
print(dmin,dmax)

# Check missing values
dataset.isna().sum()

# Standardised data
scaler = MinMaxScaler()
ds0=scaler.fit_transform(dataset)
dataset2=pd.DataFrame(ds0, columns=dataset.columns)
dataset2.head(10)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense,Bidirectional
from keras.layers import *
from keras.models import *
from tensorflow import keras
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from __future__ import print_function
from keras import backend as K
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.layers import Embedding, Dropout, Dense
from keras.layers.advanced_activations import LeakyReLU

# Splitting data
def split_dataset(data,data2):
    
    train, test = data[:17544], data[17544:]
    train2,test2 = data2[:17544], data2[17544:]
    train = np.array(np.split(train, len(train)/24))
    test = np.array(np.split(test, len(test)/24))
    train2=np.array(np.split(train2, len(train2)/24))
    test2= np.array(np.split(test2, len(test2)/24))

    return train, test,train2, test2

# Evaluate forecasts
def evaluate_forecasts(actual, predicted):

    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    print('actual.shape[0]:{}, actual.shape[1]:{}'.format(actual.shape[0], actual.shape[1]))
    return score

def summarize_scores(name, score):
    print('%s: [%.3f] \n' % (name, score))

# Set sliding-window
def sliding_window(train, sw_width=168, n_out=24, in_start=0):

    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    
    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        in_start += 24 
        
    return np.array(X), np.array(y)

# Bilstm-Attention model
def Bilstm_Attention_model(train, sw_width, in_start=0, verbose_set=0, epochs_num=20, batch_size_set=4):

    train_x, train_y = sliding_window(train, sw_width, in_start=0)
    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    
    inputs=Input(shape=(n_timesteps, n_features))
    model = Bidirectional(LSTM(200,activation='tanh', 
                                stateful=False), 
                                merge_mode='concat')(inputs)
    attention = Dense(400, activation='sigmoid', name='attention_vec')(model)#calculating Attention weights 
    model = Multiply()([model, attention])
    model = Dense(100, activation='tanh')(model) # activate function
    outputs = Dense(n_outputs)(model)
    model=Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    history=model.fit(train_x, train_y,
              epochs=epochs_num, batch_size=batch_size_set,verbose=verbose_set)
    print(model.summary())
    model.save('D:\\model and data\\BiLSTM Attention\\BiLA-AL945')
    return model, history

# Forecast
def forecast(model, pred_seq, sw_width):
   
    data = np.array(pred_seq)
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    
    input_x = data[-sw_width:, :]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    
    yhat = model.predict(input_x, verbose=0) # predict data
    yhat = yhat[0] # obtain prediction vector
    return yhat


# Evaluate model
def evaluate_model(model, train, test,test2 ,sd_width):
    
    history_fore = [x for x in train]
    predictions = list() 
    for i in range(len(test)):
        yhat_sequence = forecast(model, history_fore, sd_width)
        predictions.append(yhat_sequence)
        history_fore.append(test2[i, :]) 
    predictions = np.array(predictions) 
    for i in range (predictions.shape[0]):
        for j in range (predictions.shape[1]):
            predictions[i,j]=predictions[i,j]*(dmax-dmin)+dmin
    score = evaluate_forecasts(test[:, :,0], predictions)
    return score, predictions

def model_plot(history,epochs):

    loss = history.history['loss']
    col_name = ['loss']
    df = pd.DataFrame(columns=col_name, data=loss)
    df.to_csv('D:\\model and data\\BiLSTM Attention\\BiLA-AL945_Mloss.csv')
    epochs_range = range(epochs)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.legend(loc='upper right')
    plt.title('Train Loss')
    plt.show()

def main_run(dataset,dataset2,sw_width, name, in_start, verbose, epochs, batch_size):
    
    train, test,train2, test2= split_dataset(dataset.values, dataset2.values)#divide training and testing sets
    model,history = lstm_model(train2, sw_width, in_start, verbose_set, epochs_num, batch_size_set)#train model
    score,predictions= evaluate_model(model, train2, test,test2, sw_width)#evaluate model
    Y=predictions.reshape(-1,1)
    df = pd.DataFrame(data=Y)
    df.to_csv('D:\\model and data\\BiLSTM Attention\\BiLA-AL945_Mpred.csv')
    Y2=np.array(Y)
    df2 = pd.DataFrame({'pred': Y2.reshape(-1),
                        'true': dataset.values[17544:,0],
                       })
    R= df2.corr()
    summarize_scores(name, score)
    model_plot(history,epochs)

if __name__ == '__main__':
    
    name = 'BiLSTM-Attention'
    train, test,train2, test2= split_dataset(dataset.values, dataset2.values)
    sliding_window_width=168 
    input_sequence_start=0
    epochs_num=200 
    batch_size_set=500 
    verbose_set=0
    
    
    main_run(dataset,dataset2, sliding_window_width, name, input_sequence_start,
             verbose_set, epochs_num, batch_size_set)
