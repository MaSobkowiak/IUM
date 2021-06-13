import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow as tf

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
validate = pd.read_csv('validate.csv')

X_train = train.loc[:, train.columns != 'suicides_no']
y_train = train[['suicides_no']]
X_test = test.loc[:, train.columns != 'suicides_no']
y_test = test[['suicides_no']]

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(X_train))

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

model.summary()

model.load_weights('suicide_model.h5')

predictions = model.predict(X_test)

error = mean_squared_error(y_test, predictions)

with open('results.txt', 'a') as f:
    f.write(str(error) + "\n")

with open('results.txt', 'r') as f:
    lines = f.readlines()

fig = plt.figure(figsize=(10, 5))
chart = fig.add_subplot()

chart.set_ylabel("Mean Squared Error")
chart.set_xlabel("Build number")

x = np.arange(0, len(lines), 1)

y = [float(val) for val in lines]

plt.plot(x, y, "bo")

plt.savefig("plot.png")
