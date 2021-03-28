import argparse
import datetime
import os
import random
from pprint import pformat

import numpy as np
import torch
import yaml
from torchvision import datasets, transforms
from tqdm import tqdm

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src.attribution_methods import attribution_loader
from src.utils import utils

from src.dataset.factory import create_dataset
from src.models import utils as models_utils
from src.utils.sysutils import get_cores_count


def remove(image, attribution, replace_value, percentiles, bottom=False, gray=False):
    """
    images        : tensor of shape [C,H,W]
    attributions  : tensor of shape [H,W]
    replace_value : value to replace pixels of original image with
    percentile    : scalar between 0 and 100, inclusive. Remove percentile % of pixels
    bottom        : if true keep percentile percent(keeps top percentile percent);
                    otherwise remove 100-percentile percent(keeps bottom percentile percent)
    gray         :
    """
    modified_images = []
    masks = []
    for percentile in percentiles:
        # Convert to 1D nummpy array
        modified_image = np.copy(image)

        if gray:
            pixels_replace_threshold = int((percentile * image.size) / 300)
            if len(attribution.shape) == 3:  # gradcam gives 3 channel image
                attribution_tmp = np.array(np.ravel(np.copy(attribution[0])))
            else:  # If single channel image is passed
                attribution_tmp = np.array(np.ravel(np.copy(attribution)))
            mask = np.zeros(attribution_tmp.shape, dtype=bool)
        else:
            pixels_replace_threshold = int(percentile * image.size / 100)
            attribution_tmp = np.array(np.ravel(np.copy(attribution)))
            mask = np.zeros(attribution_tmp.shape, dtype=bool)

        if bottom:
            attribution_index = (attribution_tmp).argsort()[:pixels_replace_threshold][::-1]  # Indices of lowest values
        else:
            attribution_index = attribution_tmp.argsort()[-pixels_replace_threshold:][::-1]  # Indices of lowest values

        mask[attribution_index] = True

        if gray:
            mask = mask.reshape(image[0].shape)
            # sum = 0
            for i in range(3):  # ToDo - Dont hardcode channels
                # sum += np.count_nonzero(mask)
                modified_image[i, mask] = replace_value[i]
        else:
            mask = mask.reshape(image.shape)
            # sum = 0
            for i in range(3):  # ToDo - Dont hardcode channels
                # sum += np.count_nonzero(mask[i])
                modified_image[i, mask[i]] = replace_value[i]

        modified_images.append(modified_image)

    return modified_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/lefr_imagenet_resnet18.yml",
                        help="Configuration file to use.")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=Loader)

    perform_perturbation_analysis(cfg)


def perform_perturbation_analysis(arguments):
    """
    """

    val_data_args = dict(
        batch_size=1,
        shuffle=False
    )

    """ Setup result directory """
    outdir = arguments['outdir']
    os.makedirs(outdir, exist_ok=True)
    print('Arguments:\n{}'.format(pformat(arguments)))

    """ Set random seed throughout python"""
    random_seed = random.randint(0, 1000)
    utils.set_random_seed(random_seed=random_seed)

    """ Set device - cpu or gpu """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device - {device}')

    """ Load Model with weights(if available) """
    dataset_args = arguments['data']
    model_args = arguments['pixel_perturbation_analysis']['model']
    model: torch.nn.Module = models_utils.get_model(model_args, device, dataset_args).to(device)

    """ Load parameters for the Dataset """
    if dataset_args['dataset'] == 'ImageNet':
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
        testset = datasets.ImageNet(dataset_args['dataset_dir'], split='val', transform=transform)
    else:
        # ToDo - Make uniform api for loading different datasets. Birdsnap/Imagenet/Food101 supports needs to be added.
        dataset = create_dataset(dataset_args,
                                 val_data_args,  # Just use val_data_args as train_data_args
                                 val_data_args)  # Split doesnt matter, we use test dataset
        testset = dataset.testset

    num_samples = min(arguments['pixel_perturbation_analysis']['test_samples'], len(testset))
    print(f'Test dataset has {len(testset)} samples. We are using randomly selected {num_samples} samples for testing.')

    testset = torch.utils.data.Subset(testset, random.sample(range(0, len(testset)), num_samples))
    dataloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=get_cores_count())

    # Attribution method and percentiles at which to test.
    attribution_methods = arguments['pixel_perturbation_analysis']['attribution_methods']
    print('Running pixel perturbation analysis for: ', attribution_methods)

    # Step sizes to remove top k or bottom k
    percentiles = arguments['pixel_perturbation_analysis']['percentiles']

    # Save plots in outdir in outdir/DATASET_[train/test]/MODEL_ATTRIBUTIONMETHOD/ImageIndex.png
    model.eval()
    timestamp = datetime.datetime.now().isoformat()

    # To save sum of delta output change for each attribution method, percentile and
    # remove top and bottom percentile pixels.
    # ToDo - Use a numpy dictionary with key attribution names.
    #  Pros - Easy plotting. Load npy files without need to remember which index mapped to which attribution method.
    output_deviation_sum = np.zeros((len(attribution_methods), len(percentiles), 2), dtype=float)

    # Save results in corresponding directory
    attribution_output_dir = os.path.join(outdir, timestamp)  # E.g. outdir/timestamp/
    os.makedirs(attribution_output_dir, exist_ok=True)

    for counter, data in enumerate(tqdm(dataloader, total=num_samples)):
        if counter == num_samples:
            break
        inputs, labels = data
        inputs = inputs.to(device)
        outputs = model(inputs).detach().cpu()
        _, max_prob_indices = torch.max(outputs.data, 1)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.numpy()

        for attribution_method_index, attribution_method in enumerate(attribution_methods):

            for preprocessed_image, max_prob_index, output in zip(inputs, max_prob_indices, outputs):
                attribution_map = attribution_loader.generate_attribution(
                    model,
                    preprocessed_image.unsqueeze(0),
                    max_prob_index.to(device),
                    attribution_method
                )

                # To take absolute value for each pixel channel for each attribution method.
                attribution_map = np.max(attribution_map, axis=0)

                preprocessed_image = preprocessed_image.cpu().numpy()
                modified_images_bottom_remove = remove(preprocessed_image.copy(),
                                                       attribution_map,
                                                       replace_value=[0, 0, 0],
                                                       # Black in original image is -mean/std in preprocessed image
                                                       percentiles=percentiles,
                                                       bottom=True,
                                                       gray=True)
                modified_images_top_remove = remove(preprocessed_image.copy(),
                                                    attribution_map,
                                                    replace_value=[0, 0, 0],
                                                    # Black in original image is -mean/std in preprocessed image
                                                    percentiles=percentiles,
                                                    bottom=False,
                                                    gray=True)

                # Create a batch of all images
                modified_images_top_remove = torch.from_numpy(np.stack(modified_images_top_remove, axis=0)).to(device)
                modified_images_bottom_remove = torch.from_numpy(np.stack(modified_images_bottom_remove, axis=0)).to(
                    device)

                # Run forward pass - ToDo - Do in single pass
                output_top_q = model(modified_images_top_remove)
                output_bottom_q = model(modified_images_bottom_remove)

                output_top_q = torch.nn.functional.softmax(output_top_q, dim=1)
                output_bottom_q = torch.nn.functional.softmax(output_bottom_q, dim=1)

                output_top_q = output_top_q.detach().cpu().numpy()
                output_bottom_q = output_bottom_q.detach().cpu().numpy()

                # Get output value at max_prob_index for each percentile
                output_top_q_max_class_prob = output_top_q[:, max_prob_index]
                output_bottom_q_max_class_prob = output_bottom_q[:, max_prob_index]

                # Compute deviation from model output for original image at max_prob_index
                top_deviation = np.abs((output[max_prob_index] - output_top_q_max_class_prob) / output[max_prob_index])
                bottom_deviation = np.abs(
                    (output[max_prob_index] - output_bottom_q_max_class_prob) / output[max_prob_index])

                # Add this deviation to right dimension of matrix
                output_deviation_sum[attribution_method_index, :, 0] += top_deviation
                output_deviation_sum[attribution_method_index, :, 1] += bottom_deviation

        if counter % 500 == 499:
            # Divide output_deviation_sum each element by num_samples
            output_deviation_mean = output_deviation_sum * 100.0 / (counter + 1)

            print("\nAffect of removal of most important pixels at:-")
            for attribution_method_index, attribution_method in enumerate(attribution_methods):
                with np.printoptions(precision=3, formatter={'float': '{: 0.3f}'.format}, suppress=True,
                                     linewidth=np.inf):
                    print(attribution_method['name'].ljust(20) + ' = ',
                          np.array2string(output_deviation_mean[attribution_method_index, :, 0], separator=', '))

            print("Affect of removal of least important pixels at:-")
            for attribution_method_index, attribution_method in enumerate(attribution_methods):
                with np.printoptions(precision=3, formatter={'float': '{: 0.3f}'.format}, suppress=True,
                                     linewidth=np.inf):
                    print(attribution_method['name'].ljust(20) + ' = ',
                          np.array2string(output_deviation_mean[attribution_method_index, :, 1], separator=', '))
            print()

    # Divide output_deviation_sum each element by num_samples
    output_deviation_mean = output_deviation_sum * 100.0 / num_samples

    # Save in directory
    np.save(os.path.join(attribution_output_dir, 'pixel_perturbation.npy'), output_deviation_mean)

    with np.printoptions(precision=3, formatter={'float': '{: 0.3f}'.format}, suppress=True, linewidth=np.inf):
        print("Affect of removal of most important pixels at:- \npercentiles ", percentiles)
        for ind, attr in enumerate(attribution_methods):
            print(attr['name'].ljust(20), output_deviation_mean[ind, :, 0])
    print()
    with np.printoptions(precision=3, formatter={'float': '{: 0.3f}'.format}, suppress=True, linewidth=np.inf):
        print("Affect of removal of least important pixels at:- \npercentiles ", percentiles)
        for ind, attr in enumerate(attribution_methods):
            print(attr['name'].ljust(20), output_deviation_mean[ind, :, 1])


if __name__ == '__main__':
    main()
