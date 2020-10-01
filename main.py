import os
from train import train
from predict import predict
import argparse

load_dir = 'train_data'
if not os.path.exists(load_dir):
    raise FileNotFoundError("Run setup.py first")

if __name__ == "__main__":

    # Construct the argument parser
    parser = argparse.ArgumentParser()

    # Add a mutually exclusive group as training and prediction
    # cannot happen at the same time
    group = parser.add_mutually_exclusive_group(required=True)

    # Set 1000 epochs as default for training
    group.add_argument('-t', '--train', type=int, default=1000)

    # The predict argument will take a csv filepath as input
    group.add_argument('-p', '--predict', type=str)
    args = vars(parser.parse_args())

    if args['predict'] is not None:
        predict_filepath = args['predict']
        predict(predict_filepath)

    else:
        train_epochs = args['train']
        train(int(train_epochs))
