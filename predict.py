# from random import random
import os
import pandas as pd
from tensorflow import keras
import plot

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


def predict():
    valid_data = _load('valid')

    valid_labels = valid_data.pop(score)

    model = keras.models.load_model('model.h5')

    prediction = model.predict(valid_data).flatten()

    plot.plot_prediction(valid_labels, prediction)

    plot.plot_error(valid_labels, prediction)


if __name__ == "__main__":
    predict()
