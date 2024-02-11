# %% [markdown]
# <h3> SUBMISSION: Proyek Kedua : Membuat Model Machine Learning dengan Data Time Series <h3>
# <h4> Delvin Fachrizky

# %%
# MENYIAPKAN LIBRARY YANG AKAN DIGUNAKAN PADA PROJECT KALI INI
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout

# %%
# MENYIAPKAN DATASET
dtfrm = pd.read_csv("datasets.csv")
# MELIHAT 5 DATA TERATAS
dtfrm.head()

# %%
# MELIHAT 5 DATA TERAKHIR
dtfrm.tail()

# %%
# MELIHAT INFORMASI DARI DATASET YANG TELAH DIIMPORT
dtfrm.info()

# %%
# MELIHAT VALUE DARI KOLOM Country
dtfrm['Country'].value_counts()

# %%
# MENGAMBIL DATA DARI TAHUN 1945 HINGGA 2013
dtfrm['dt'] = pd.to_datetime(dtfrm['dt'])
get_data = (dtfrm['dt'] > '1945-01-01') & (dtfrm['dt'] <= '2013-09-01')
dtfrm.loc[get_data]

# %%
# MENGAMBIL DATA NEGARA "CHINA" DARI KOLOM DATASET YANG TELAH ADA
dtfrm = dtfrm.loc[dtfrm['Country'].isin(['China'])]
display(dtfrm)

# %%
# MELAKUKAN RESET INDEX DARI DATAFRAME DAN DROP PADA KOLOM COUNTRY
dtfrm.drop(['Country'], axis=1, inplace=True)
dtfrm.reset_index(drop=True)

# %%
# MELAKUKAN CEK TERHADAP DATA YANG MEMILIKI NILAI NULL
dtfrm.isnull().sum()

# %%
# MEMBERSIHKAN DATA YANG MEMILIKI NILAI/VALUE NULL
dtfrm.dropna(subset=['AverageTemperature'], inplace=True)
dtfrm.dropna(subset=['AverageTemperatureUncertainty'], inplace=True)
dtfrm.isnull().sum()

# %%
# MELAKUKAN PLOTING UNTUK MASING-MASING KOLOM
dtfrm_plot = dtfrm
dtfrm_plot[dtfrm_plot.columns.to_list()].plot(subplots=True, figsize=(15, 9))
plt.show()

# %%
# PLOT WAKTU DAN TEMPERATUR
date_time = dtfrm['dt'].values
temperature = dtfrm['AverageTemperature'].values

date_time = np.array(date_time)
temperature = np.array(temperature)

plt.figure(figsize=(15,9))
plt.plot(date_time, temperature)

plt.title('Average Temperature', fontsize=20)
plt.ylabel('Temperature')
plt.xlabel('DateTime')

dtfrm.dtypes

# %%
# MELAKUKAN SPLIT DATASET DENGAN PERBANDINGAN 80:20 TRAINING/TESTING
X_train, X_test, y_train, y_test = train_test_split(temperature, date_time, train_size=0.8, test_size = 0.2, shuffle = False )
print('Data Train : ',len(X_train))
print('Data Validation : ',len(X_test))

# %%
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  series = tf.expand_dims(series, axis=-1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1, shift=1, drop_remainder = True)
  ds = ds.flat_map(lambda w: w.batch(window_size + 1))
  ds = ds.shuffle(shuffle_buffer)
  ds = ds.map(lambda w: (w[:-1], w[-1:]))
  return ds.batch(batch_size).prefetch(1)

# %%
# SEQUENTIAL MODELING
tf.keras.backend.set_floatx('float64')

train_set = windowed_dataset(X_train, window_size=64, batch_size=200, shuffle_buffer=1000)
val_set = windowed_dataset(X_test, window_size=64, batch_size=200, shuffle_buffer=1000)

model = Sequential([
    Bidirectional(LSTM(60, return_sequences=True)),
    Bidirectional(LSTM(60)),
    Dense(30, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1),
])

# %%
# MENGHITUNG MAE 10% DAN IMPLEMENTASI DARI CALLBACK
Mae = (dtfrm['AverageTemperature'].max() - dtfrm['AverageTemperature'].min()) * 10/100
print(Mae)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae')<2.4 and logs.get('val_mae')<2.4):
      print("\nMAE dari model < 10% skala data")
      self.model.stop_training = True
callbacks = myCallback()

# %%
# LEARNING RATE SGD OPTIMIZER
optimizer = tf.keras.optimizers.SGD(lr=1.0000e-04, momentum=0.9)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])

history = model.fit(train_set, epochs=100, validation_data = val_set, callbacks=[callbacks])

# %%
# MEMBUAT PLOT AKURASI MODEL
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Akurasi Model')
plt.ylabel('Mae')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# %%
# MEMBUAT PLOT LOSS MODEL
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


