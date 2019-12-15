import os
from train import train
from predict import predict

load_dir = 'train_data'
if not os.path.exists(load_dir):
    raise FileNotFoundError("Run setup.py first")

if __name__ == "__main__":
    last_arg = os.sys.argv[-1]
    if last_arg in ['p', 'predict']:
        predict()
    elif last_arg.isdigit() and os.sys.argv[-2] in ['t', 'train']:
        train(int(last_arg))
    elif last_arg in ['t', 'train']:
        train()
    else:
        raise TypeError("Use p(redict) or t(rain) [epochs]")
