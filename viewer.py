"""
Code for viewing the data.
"""

import os
import matplotlib.pyplot
import matplotlib.cm
import numpy as np


def show_example(image, label):
    """
    Views an image an its corresponding label.

    :param image: The image array.
    :type image: np.ndarray
    :param label: The label array.
    :type label: np.ndarray
    """
    mappable = matplotlib.cm.ScalarMappable(cmap='inferno')
    mappable.set_clim(vmin=label.min(), vmax=label.max())
    label_heatmap = mappable.to_rgba(label).astype(np.float32)
    matplotlib.pyplot.subplot(1, 2, 1)
    matplotlib.pyplot.imshow(image)
    matplotlib.pyplot.subplot(1, 2, 2)
    matplotlib.pyplot.imshow(label_heatmap)
    matplotlib.pyplot.show()

dataset_directory = '/Users/golmschenk/Desktop/world_expo_200741'
images = np.load(os.path.join(dataset_directory, 'images.npy'), mmap_mode='r')
labels = np.load(os.path.join(dataset_directory, 'labels.npy'), mmap_mode='r')
show_example(images[0], labels[0])
show_example(images[1], labels[1])
show_example(images[-1], labels[-1])
show_example(images[-2], labels[-2])
