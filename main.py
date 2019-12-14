# from random import random
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import plot

load_dir = 'train_data'
if not os.path.exists(load_dir):
    raise FileNotFoundError("Run setup.py first")

score = 'vote_average'


def load(file_name):
    return pd.read_csv(
        f'{load_dir}/{file_name}.csv',
        index_col=False,
        skipinitialspace=True
    )


test_data = load('test')
train_data = load('train')
valid_data = load('valid')

train_labels = train_data.pop(score)
test_labels = test_data.pop(score)
valid_labels = valid_data.pop(score)


def build_model():
    model = keras.Sequential([
        layers.Dense(64,
                     activation=tf.nn.relu,
                     input_shape=[len(train_data.keys())]
                     ),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()

model.summary()

EPOCHS = 1000


class NetworkCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if (epoch == 0):
            self.pbar = tqdm(range(EPOCHS))
        self.pbar.update(1)
        if (epoch == EPOCHS - 1):
            self.pbar.close()


history = model.fit(
    train_data,
    train_labels,
    batch_size=100,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[NetworkCallback()]
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plot.plot_history(history)

loss, mae, mse = model.evaluate(test_data, test_labels, verbose=1)

print("Testing set Mean Abs Error: {:5.2f} Score".format(mae))

prediction = model.predict(valid_data).flatten()

plot.plot_prediction(valid_labels, prediction)

plot.plot_error(valid_labels, prediction)
