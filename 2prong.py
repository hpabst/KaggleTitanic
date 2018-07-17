import matplotlib.pyplot as plt
import pandas as pd
import math
from keras import backend, Input
from keras.layers import concatenate, Dense, BatchNormalization

from keras.utils import plot_model

pd.set_option("mode.chaned_assignment", None)

def smooth_curve(points, factor=0.75):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def set_plot_history_data(ax, history, which_graph):
    if which_graph == "acc":
        train = smooth_curve(history.history['acc'])
        valid = smooth_curve(history.history['val_acc'])
    if which_graph == "loss":
        train = smooth_curve(history.history["loss"])
        valid = smooth_curve(history.history["val_loss"])

    plt.xkcd()
    epochs = range(1, len(train) + 1)
    trim = 5
    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', label=('Training'))
    ax.plot(epochs[trim:], train[trim:], 'dodgerblue', linewidth=15, alpha=0.1)

    ax.plot(epochs[trim:], valid[trim:], 'g', label=('Validation'))
    ax.plot(epochs[trim:], valid[trim:], 'g', linewidth=15, alpha=0.1)

def get_max_validation_accuraccy(history):
    validation = smooth_curve(history.history['val_acc'])
    ymax = max(validation)
    return 'Max validation accuracy = ' + str(round(ymax, 3)*100) + '%'

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [5,2]})
    set_plot_history_data(ax1, history, 'acc')
    set_plot_history_data(ax2, history, 'loss')
    #Accuracy graph
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(bottom=0.5, top=1)
    ax1.legend(loc='lower right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.xaxis.set_ticks_position('none')
    ax1.spines['bottom'].set_visible(False)

    #max accuracy text
    plt.text(0.97,
             0.97,
             get_max_validation_accuraccy(history),
             horizontalalignment='right',
             verticalalignment='top',
             transform=ax1.transAxes,
             fontsize=12)

    #Loss graph
    ax2.set_ylabel('Loss')
    ax2.set_yticks([])
    ax2.plot(legend=False)
    ax2.set_xlabel('Epochs')
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()