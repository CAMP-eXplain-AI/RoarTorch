import numpy as np


def remove(image, attribution, mean, percentile, keep=False, gray=False):
    """
    images       : tensor of shape [C,H,W]
    attributions : tensor of shape [H,W]
    mean         : mean of dataset
    percentile   : scalar between 0 and 100, inclusive
    keep         : if true keep q percent; otherwise remove q percent
    gray         :
    """

    # Convert to 1D numpy array
    modified_image = np.copy(image)
    if gray:
        pixels_replace_threshold = int(percentile * (image.size / 3) / 100)
        attribution_tmp = np.array(np.ravel(np.copy(attribution[0])))
        mask = np.zeros(attribution_tmp.shape, dtype=bool)
    else:
        pixels_replace_threshold = int(percentile * image.size / 100)
        attribution_tmp = np.array(np.ravel(np.copy(attribution)))
        mask = np.zeros(attribution_tmp.shape, dtype=bool)

    if keep:
        # Todo - KAR might need more testing, since we didnt use it as it is less reliable.
        lower_attribution_index = (attribution_tmp).argsort()[:pixels_replace_threshold][
                                  ::-1]  # Indices of lowest values
        mask[lower_attribution_index] = True
    else:
        higher_attribution_index = attribution_tmp.argsort()[-pixels_replace_threshold:][::-1]
        mask[higher_attribution_index] = True

    if gray:
        mask = mask.reshape(image[0].shape)
        for i in range(3):
            modified_image[i, mask] = mean[i]
    else:
        mask = mask.reshape(image.shape)
        for i in range(3):
            modified_image[i, mask[i]] = mean[i]

    return modified_image


def validate_configuration(cfg: dict,
                           validate_dataset: bool = True,
                           validate_attribution_methods: bool = False):
    """ Validates configuration for valid datasets and valid attribution method names """

    if validate_dataset:
        valid_datasets = ['CIFAR10', 'Food101', 'Birdsnap', 'ImageNet']
        assert cfg['data']['dataset'] in valid_datasets, \
            'dataset {} not supported, valid choices: {}'.format(cfg['dataset'], valid_datasets)

    if validate_attribution_methods:
        attribution_methods = cfg['extract_cams']['attribution_methods']
        attribution_method_names = []
        for attribution_method in attribution_methods:
            attribution_method_names.append(attribution_method['name'])
        assert len(attribution_method_names) == len(set(attribution_method_names)), \
            'All attribute method names should be unique.'
