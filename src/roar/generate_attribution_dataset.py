import argparse
import logging
import os
import random
from collections import defaultdict
from pprint import pformat

import numpy as np
import skimage.io
import torch
import yaml
from tqdm import tqdm

from src.attribution_methods import attribution_loader

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src.roar import roar_core
from src.utils import logger, utils
from src.dataset.factory import create_dataset, MAP_DATASET_TO_ENUM
from src.models import utils as model_utils


def convert_float_to_percentiled_3channel_image(arr):
    """
    Returns image with color pixel intensity as [ 0,  1,  2,  3, ...., 255], where pixels
    :param percentiles:
    :param arr: NDarray.
    """
    if arr.max() == arr.min():
        return np.zeros(arr.shape).astype('uint8')

    percentiled_3channel_image = ((arr - arr.min()) / (arr.max() - arr.min()))
    percentiled_3channel_image = np.uint8(percentiled_3channel_image * 255)
    return percentiled_3channel_image


def dump_saliency_data():
    """
    Main Pipeline for training and cross-validation.
    """

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="config/cifar10_resnet8.yml", help="Configuration file to use.")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=Loader)

    roar_core.validate_configuration(cfg, validate_attribution_methods=True)

    # Common Configuration
    dataset_args = cfg['data']

    train_data_args = dict(
        batch_size=4,
        shuffle=False,
        enable_augmentation=False,
    )

    assert not train_data_args['enable_augmentation'], \
        'Augmentation of dataset should be disabled for generating dataset'

    val_data_args = dict(
        batch_size=4,
        shuffle=False,
    )

    # Shuffling should be off
    assert not val_data_args['shuffle']

    arguments = dict(
        dataset_args=dataset_args,
        train_data_args=train_data_args,
        val_data_args=val_data_args,
        model_args=cfg['extract_cams']['model'],
        outdir=cfg['outdir'],

    )

    """ Setup result directory """
    outdir = os.path.join(arguments.get("outdir"), 'extract_cams')
    logger.init(outdir, filename_prefix='extract_cams', log_level=logging.INFO)  # keep logs at root dir.
    logger.info('Arguments:\n{}'.format(pformat(arguments)))

    """ Set random seed throughout python"""
    utils.set_random_seed(random_seed=random.randint(0, 1000))

    """ Set device - cpu or gpu """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device - {device}')

    """ Load parameters for the Dataset """
    dataset = create_dataset(dataset_args, train_data_args, val_data_args)

    """ Sample and View the inputs to model """
    dataset.debug()

    dataloaders = [dataset.train_dataloader, dataset.validation_dataloader, dataset.test_dataloader]
    dataset_modes = ['train', 'validation', 'test']

    # Some datasets do not have same image size and thus we need to crop a certain area to compute attribution maps
    # Since we want to use same input image for retraining of model with different attribution maps, care must be taken
    # that transform applied gives same input image. Such as center crop is fine, but RandomCrop/RandomScale are not.
    save_input_images = True
    save_attribution = True

    assert save_input_images or save_attribution, 'Either save input images or save attribution flag should be enabled.'
    attribution_methods = cfg['extract_cams']['attribution_methods']
    logger.info(f'Computing attribution maps for {attribution_methods}')

    # Save attribution maps in ./outdir/[train, validation, test]/[input/AttributionName]/Class/ImageIndex.png
    for attribution_method in attribution_methods:
        for dataloader, dataset_mode in zip(dataloaders, dataset_modes):

            """ Load Model with weights(if available) """
            model: torch.nn.Module = model_utils.get_model(
                arguments.get('model_args'), device, arguments['dataset_args']
            ).to(device)

            """ Create cropped dataset and attributions directories """
            # Need to save cropped dataset once - Although its is not optimum to have another copy of a dataset, we
            # still preferred this due to simpler design of having parallel attribution and images dataset.
            if save_input_images:
                # The Classification Dataset(CIFAR/Birdsnap) are written in torchvision.datasets.ImageFolder format.
                images_output_dirs = [os.path.join(outdir, f'{dataset_mode}/input/', str(cls))
                                      for cls in dataset.classes]
                [os.makedirs(dir, exist_ok=True) for dir in images_output_dirs]

            counter = 0
            # Create labelled attribution folder
            attribution_output_dirs = [os.path.join(outdir, f'{dataset_mode}/{attribution_method["name"]}', str(cls))
                                       for cls in dataset.classes]
            [os.makedirs(dir, exist_ok=True) for dir in attribution_output_dirs]
            counters = defaultdict(int)

            """ Thank god, finally let CAM extraction begin """
            logger.info(f"Generating images and attribution for {attribution_method} in {dataset_mode} split.")
            model.eval()
            for i, data in enumerate(tqdm(dataloader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, max_prob_indices = torch.max(outputs.data, 1)

                # ToDo: Add support for method that can compute attributions for batch
                # TODo - Check width and height
                for preprocessed_image, max_prob_index, label in zip(inputs, max_prob_indices, labels):
                    """ Save CAMS and input images."""
                    if save_input_images:
                        # Denormalize the image and save
                        rgb_image = dataset.denormalization_transform(preprocessed_image.cpu())
                        rgb_image = (torch.clamp(rgb_image, 0.0, 1.0).numpy() * 255.0).astype('uint8')
                        skimage.io.imsave(f'{images_output_dirs[label]}/'
                                          f'{str(counters[label.item()]).zfill(5)}.png',
                                          rgb_image.transpose(1, 2, 0),
                                          check_contrast=False)

                    if save_attribution:
                        attribution_map = attribution_loader.generate_attribution(
                            model,
                            preprocessed_image.unsqueeze(0),
                            max_prob_index,
                            attribution_method
                        )

                        # Save in attribution_output_dir as a uint8 image
                        percentiled_image = convert_float_to_percentiled_3channel_image(attribution_map)
                        skimage.io.imsave(f'{attribution_output_dirs[label]}/'
                                          f'{str(counters[label.item()]).zfill(5)}.png',
                                          percentiled_image.transpose(1, 2, 0),
                                          check_contrast=False)
                    counters[label.item()] += 1

        save_input_images = False  # No need to resave input images for next attribution method


if __name__ == '__main__':
    dump_saliency_data()
