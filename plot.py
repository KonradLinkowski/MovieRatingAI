import os
import pandas as pd
import matplotlib.pyplot as plt

plot_dir = 'plots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def plot_history(history, file_name='history.png'):
    plt.clf()
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Score]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 10])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$Score^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])
    plt.savefig(plot_dir + file_name)


def plot_prediction(valid, prediction, file_name='prediction.png'):
    plt.clf()
    plt.scatter(valid, prediction)
    plt.xlabel('Real values')
    plt.ylabel('vote_average prediction')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])

    plt.savefig(plot_dir + file_name)


def plot_error(labels, prediction, file_name='error.png'):
    plt.clf()
    error = prediction - labels
    plt.hist(error, bins=25)
    plt.xlabel('Prediction error [vote_average]')
    _ = plt.ylabel('Count')
    plt.savefig(plot_dir + file_name)
