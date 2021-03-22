import importlib

import numpy as np


def generate_attribution(model, preprocessed_image, label, attribution_method):
    """
    Loads and calls attribution method function and validates its output.

    :param model:  torch.nn.Module to be passed to attribution method
    :param preprocessed_image: The preprocessed image fed to the model
    :param label: The label for which attribution maps to compute
    :param attribution_method: A dictionary. See config/cifar10_resnet8.yaml for dict keys to add.
    :return: A 3xHxW attribution map.
    """
    module_name, method = attribution_method['method'].rsplit('.', 1)  # p is module(filename), m is Class Name
    module_obj = importlib.import_module(module_name)
    attribution_func = getattr(module_obj, method)

    attribution_value = attribution_func(model, preprocessed_image, label, **attribution_method['kwargs'])

    assert type(attribution_value) == np.ndarray
    assert attribution_value.shape.__len__() == 3, \
        f'{attribution_method} return value should be 3 dimensional.'
    assert attribution_value.shape[0] == 3, \
        f'{attribution_method} return value should have dimension 3xHxW, found {attribution_value.shape}.'
    return attribution_value
