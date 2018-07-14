"""Useful utility function."""

# Standard imports
import math
import logging

# Dependency imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(flat_labels, flat_preds, target_names,
                          title='Confusion matrix', cmap=plt.cm.Blues, # pylint: disable=E1101
                          save_path=None, percent=True):

    """Plot confusion matrix."""

    bin_n = np.bincount(flat_labels.astype(np.int64))
    counts = np.nonzero(bin_n)[0]
    print('Label distribution', list(zip(counts, bin_n[counts])))

    c_m = confusion_matrix(
        flat_labels,
        flat_preds
    ).astype(np.float32) # pylint: disable=E1101

    if percent:
        for i in range(c_m.shape[0]):
            c_m[i] = [np.round_(100 * _cm / np.sum(c_m[i]), 2) for _cm in c_m[i]]

    plt.figure(figsize=(8, 8))
    plt.grid(False)
    plt.imshow(c_m, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = c_m.shape

    for x_val in range(width):
        for y_val in range(height):
            plt.annotate(str(c_m[x_val][y_val]), xy=(y_val, x_val),
                         horizontalalignment='center',
                         verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path is not None:
        print('Confusion Matrix graph saved to ' + save_path)
        plt.savefig(save_path)

def pim(images, titles=None, cols=0, cmap=None, interpolation='none'):
    """Plot Image.

    Display one or more images in one row without blocking.

    Input :
        images : list of images as numpy arrays
        titles : list of title strings for each image. Default : None
        cmap : color map for showing image. Default : gray

    Output :
        Plot all images in one row with plt.show(block=False)
    """

    if titles is None:
        titles = []

    if not isinstance(images, list) and not isinstance(images, np.ndarray):
        logging.warning('Input must be list or numpy array')
        return None

    n_im = len(images)

    if cols < 1:
        cols = len(images)
    rows = int(math.ceil(n_im / cols))

    _gridspace = gridspec.GridSpec(rows, cols)

    fheight = int(math.ceil(10 * rows / cols))

    fig = plt.figure(figsize=(20, fheight))
    for i_im in range(n_im):
        axis = fig.add_subplot(_gridspace[i_im])
        axis.imshow(images[i_im], cmap=cmap, interpolation=interpolation)

        try:
            axis.set_title(titles[i_im] + ' ' + str(images[i_im].shape))
        except IndexError:
            axis.set_title('Image ' + str(images[i_im].shape))

    _gridspace.tight_layout(fig)
    plt.show()
