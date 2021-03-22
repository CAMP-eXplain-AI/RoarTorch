import importlib

import torch
import torchvision

from src.utils import logger


def convert_sample_to_batch(input):
    return input.unsqueeze(0)


def print_learnable_params(net: torch.nn.Module, recurse=True):
    params = list(net.parameters(), recurse)
    print(params)


def num_flat_features(x):
    """
    Computes number of features in a single sample of a batch
    :param x: The batch of inputs/features
    :return: Number of features in a single sample of a batch.
    """
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def append_linear_layer_transform(model, num_of_classes):
    if type(model) == torchvision.models.resnet.ResNet:
        if num_of_classes == model.fc.out_features:  # Do not replace anything if output classes is same as requested.
            return model
        else:
            logger.warn(f'Model appended another FC layer transforming number of output classes to {num_of_classes}')
            model = torch.nn.Sequential(model,
                                        torch.nn.Linear(in_features=model.fc.out_features,
                                                        out_features=num_of_classes,
                                                        bias=True))

    return model


def get_model(model_args: dict, device, dataset_args) -> torch.nn.Module:
    """
    Returns module object for relative module name
    :param model_args: dict containing model qualified name, weight path if available and model constructor params.
    :return: Module object for model name.
    """
    module_name, m = model_args['arch'].rsplit('.', 1)
    module_obj = importlib.import_module(module_name)
    model = getattr(module_obj, m)
    if model_args['constructor_args'] is not None:
        model: torch.nn.Module = model(**model_args['constructor_args'])
    else:
        model: torch.nn.Module = model()

    # ToDo Fix below code
    if model_args.get('transformer') is not None:
        labels_count = dataset_args.get('labels_count', None)  # ToDo - Only one check for label_count.
        if not labels_count:
            labels_count = dataset_args['name'].value['labels_count']
        model = model_args.get('model_transformer')(model, labels_count)

    if model_args['pretrained_weights'] is not None:
        print('Loading model weights from ', model_args['pretrained_weights'])
        model.load_state_dict(torch.load(model_args['pretrained_weights'], map_location=device), strict=True)
    return model
