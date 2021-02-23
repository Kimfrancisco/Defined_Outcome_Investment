#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def rmse(y,y_hat):
    from sklearn.metrics import mean_squared_error
    import math
    return math.sqrt(mean_squared_error(y,y_hat))


# In[2]:


wti = pd.read_csv(r'/home/gyeongho/Desktop/Project Lab/dataset(monthly)/wti.csv')
wti = wti.iloc[:240,1]
wti_original = wti
wti_mean = wti.mean()
wti_std = wti.std()
wti = (wti-wti.mean())/wti.std()
wti.shape


# In[3]:


gold = pd.read_csv(r'/home/gyeongho/Desktop/Project Lab/dataset(monthly)/gold.csv')
gold = np.array(gold['Open'])
gold = (gold-gold.mean())/gold.std()
gold.shape


# In[4]:


usdollarindex = pd.read_csv(r'/home/gyeongho/Desktop/Project Lab/dataset(monthly)/usdollarindex.csv')
usdollarindex = usdollarindex['Open']
usdollarindex = np.array(usdollarindex[:240])
usdollarindex = (usdollarindex-usdollarindex.mean())/usdollarindex.std()
usdollarindex.shape


# In[5]:


petroleum = pd.read_csv(r'/home/gyeongho/Desktop/Project Lab/dataset(monthly)/US Import Price Index_Petroleum oils and oils from bituminous minerals, crude.csv')
petroleum = petroleum.iloc[:240,1]
petroleum = np.array(petroleum)
petroleum = (petroleum-petroleum.mean())/petroleum.std()
petroleum.shape


# In[6]:


production = pd.read_csv(r'/home/gyeongho/Desktop/Project Lab/dataset(monthly)/OPEC_Total_Spare_Crude_Oil_Production_Capacity.csv', header=None)
production = production.iloc[:240,1]
production = np.flipud(production)
production = (production-production.mean())/production.std()
production.shape


# In[7]:


exports = pd.read_csv(r'/home/gyeongho/Desktop/Project Lab/dataset(monthly)/U.S._Exports_of_Crude_Oil.csv', header=None)
exports = exports.iloc[:, 1]
exports = np.flipud(exports)
exports = (exports-exports.mean())/exports.std()
exports.shape


# In[8]:


commodity = pd.read_csv(r'/home/gyeongho/Desktop/Project Lab/dataset(monthly)/CoreCommodityCRBtotalreturn.csv')
commodity = commodity.iloc[1:, 1]
commodity = np.flipud(commodity)
commodity = (commodity-commodity.mean())/commodity.std()
commodity.shape


# In[9]:


interest = pd.read_csv(r'/home/gyeongho/Desktop/Project Lab/dataset(monthly)/10 yr interest treasury yield.csv')
interest = np.array(interest['Open'])
interest = (interest-interest.mean())/interest.std()
interest.shape


# In[10]:


data = [wti, gold, usdollarindex, petroleum, production, exports, commodity, interest, wti]
data = np.column_stack(data)


# In[11]:


data.shape


# In[12]:


def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(0, len(sequences)):
        end_index = i+n_steps_in
        out_end_index = end_index+n_steps_out-1
        
        if out_end_index>len(sequences):
            break
        
        seq_x = sequences[i:end_index,:-1]
        seq_y = sequences[end_index-1:out_end_index, -1]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)


# In[13]:


n_features = data.shape[1] - 1

n_steps_in = 3          #lookback 수
n_steps_out = 1      #future의 몇개를 예측?

X, y = split_sequences(data, n_steps_in=n_steps_in, n_steps_out=n_steps_out)
train_length = np.int(0.9*X.shape[0])

X_train = X[:train_length]
X_test = X[train_length:]
y_train = y[:train_length]
y_test = y[train_length:]


# In[14]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[15]:


print(y.shape)
print(y_train.shape)
print(y_test.shape)


# In[16]:


from keras.models import Sequential
from keras import optimizers
from keras.layers import LSTM, Dense, Dropout
from keras.utils import multi_gpu_model
from keras.backend import tensorflow_backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)


# In[18]:


model = Sequential()
model.add(LSTM(units=128, activation='relu', input_shape=(n_steps_in, n_features), return_sequences=True))
model.add(LSTM(units=32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse', metrics=['mape'])


# In[19]:


#####EARLY STOPPAGE 사용

history = model.fit(X,y, epochs=10000, validation_split=0.2, verbose=False, callbacks=[es, checkpoint])


# In[20]:


history = history.history


# In[21]:


plt.plot(range(0,len(history['val_loss'])), history['val_loss'])


# In[17]:


from keras.models import load_model

model = load_model('best_model.h5')


# In[26]:


plt.rcParams["figure.figsize"] = (20,10)
y_real = (y*wti_std)+wti_mean
plt.plot(y_real)
#
y_hat_all = model.predict(X)
y_hat_all = (y_hat_all*wti_std)+wti_mean

plt.plot(y_hat_all, c='grey')
plt.show()

print(mean_absolute_error(y_real, y_hat_all))
print(mean_squared_error(y_real, y_hat_all))
print(rmse(y_real, y_hat_all))


# In[27]:


plt.rcParams["figure.figsize"] = (20,10)

y_test_real = (y_test*wti_std)+wti_mean
plt.plot(y_test_real)

y_hat = model.predict(X_test)
print(mean_squared_error(y_test, y_hat))
y_hat = (y_hat*wti_std)+wti_mean
plt.plot(y_hat, c='grey')

print(mean_absolute_error(y_test, y_hat))
print(mean_squared_error(y_test, y_hat))
print(rmse(y_test, y_hat))


# In[ ]:




