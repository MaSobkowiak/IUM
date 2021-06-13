import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

EPOCHS = int(10)
BATCH_SIZE = int(5)

train = pd.read_csv('train.csv')
validate = pd.read_csv('validate.csv')
test = pd.read_csv('test.csv')

# podział train set
X_train = train.loc[:, train.columns != 'suicides_no']
y_train = train[['suicides_no']]
X_test = test.loc[:, train.columns != 'suicides_no']
y_test = test[['suicides_no']]

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(X_train))

first = np.array(X_train[:1])
with np.printoptions(precision=2, suppress=True):
    print('First example:', first)
    print()
    print('Normalized:', normalizer(first).numpy())

model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])
model.predict(X_train[:10])

# Compile model
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Train model
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2)

model.save_weights('suicide_model.h5')

test_results = {}

test_results['model'] = model.evaluate(
    X_test, y_test, verbose=0)

test_predictions = model.predict(X_test).flatten()

predictions = model.predict(X_test)
pd.DataFrame(predictions).to_csv('results.csv')
model.summary()
