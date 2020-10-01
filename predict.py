# from random import random
import os
import pandas as pd
from tensorflow import keras
import plot
import argparse

load_dir = 'train_data'
if not os.path.exists(load_dir):
    raise FileNotFoundError("Run setup.py first")

score = 'vote_average'


def _load(file_path):

    # Ensure that the input file is a csv
    if os.path.splitext(file_path)[1] != '.csv':
        raise TypeError('Input file type must be csv')

    return pd.read_csv(
        file_path,
        index_col=False,
        skipinitialspace=True
    )


def predict(file_path):
    valid_data = _load(file_path)

    valid_labels = valid_data.pop(score)

    model = keras.models.load_model('model.h5')

    prediction = model.predict(valid_data).flatten()

    print('Prediction Output:', prediction)

    plot.plot_prediction(valid_labels, prediction)

    plot.plot_error(valid_labels, prediction)


if __name__ == "__main__":
    # Construct the argument parser
    parser = argparse.ArgumentParser()

    # Set path argument with a default validation csv file
    parser.add_argument('-p', '--path', type=str, default='train_data/valid.csv')
    args = vars(parser.parse_args())

    predict(args['path'])
