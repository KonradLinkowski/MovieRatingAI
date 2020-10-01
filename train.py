import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import plot
import argparse

load_dir = 'train_data'
if not os.path.exists(load_dir):
    raise FileNotFoundError("Run setup.py first")

score = 'vote_average'


def _load(file_name):
    return pd.read_csv(
        f'{load_dir}/{file_name}.csv',
        index_col=False,
        skipinitialspace=True
    )


test_data = _load('test')
train_data = _load('train')

train_labels = train_data.pop(score)
test_labels = test_data.pop(score)


def _build_model():
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


class NetworkCallback(keras.callbacks.Callback):
    def __init__(self, epochs):
        self.EPOCHS = epochs

    def on_epoch_end(self, epoch, logs):
        if (epoch == 0):
            self.pbar = tqdm(range(self.EPOCHS))
        self.pbar.update(1)
        if (epoch == self.EPOCHS - 1):
            self.pbar.close()


def train(EPOCHS=1000):
    model = _build_model()
    model.summary()

    history = model.fit(
        train_data,
        train_labels,
        batch_size=100,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=0,
        callbacks=[NetworkCallback(EPOCHS)]
    )

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    plot.plot_history(history)

    loss, mae, mse = model.evaluate(test_data, test_labels, verbose=1)

    print("Testing set Mean Abs Error: {:5.2f} Score".format(mae))

    model.save('model.h5')


if __name__ == "__main__":
    # Construct the argument parser
    parser = argparse.ArgumentParser()

    # Set epochs argument with default 1000
    parser.add_argument("-e", "--epochs", type=int, default=1000)
    args = vars(parser.parse_args())

    train(args['epochs'])
