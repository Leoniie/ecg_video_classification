from vis.visualization import visualize_activation, visualize_saliency
from vis.utils import utils
from keras import activations

from matplotlib import pyplot as plt
import numpy as np


def visualize(model, layername, validdata, validlabel):
    """
    :type model: keras model object
    :type layername: string of final dense layer
    :type validdata: X data sample  for validation
    :type validlabel: y label sample for validation
    """
    # Utility to search for layer index by name.
    # Alternatively we can specify this as -1 since it corresponds to the last layer.
    layer_idx = utils.find_layer_idx(model, layername)

    # Swap softmax with linear
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    # This is the output node we want to maximize.
    filter_idx = 0
    img = visualize_activation(model, layer_idx, filter_indices=filter_idx)
    plt.imshow(img[..., 0])
    plt.show()

    for class_idx in np.arange(2):
        indices = np.where(validlabel[:, class_idx] == 1.)[0]
        idx = indices[0]

        f, ax = plt.subplots(1, 4)
        ax[0].imshow(validdata[idx][..., 0])

        for i, modifier in enumerate([None, 'guided', 'relu']):
            grads = visualize_saliency(model, layer_idx, filter_indices=class_idx,
                                       seed_input=validdata[idx], backprop_modifier=modifier)
            if modifier is None:
                modifier = 'vanilla'
            ax[i + 1].set_title(modifier)
            ax[i + 1].imshow(grads, cmap='jet')
    plt.show()

    return model
