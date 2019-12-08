import os
import pandas as pd

load_dir = 'train_data'
if not os.path.exists(load_dir):
    raise FileNotFoundError("Run setup.py first")


def load(file_name):
    return pd.read_csv(f'{load_dir}/{file_name}.csv')


test_data = load('test')
train_data = load('train')
valid_data = load('valid')
