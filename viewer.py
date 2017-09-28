"""
Code for viewing the data.
"""

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

