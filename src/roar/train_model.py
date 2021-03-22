import argparse
import logging
import os
import random
import time
from pprint import pformat

import torch
import yaml
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src.roar import roar_core
from src.utils import utils
from src.dataset.factory import create_dataset
from src.loss import utils as loss_utils
from src.models import utils as models_utils
from src.optimizer import utils as optimizer_utils
from src.utils import logger
from src.utils.sysutils import is_debug_mode
from src.utils.tensorboard_writer import initialize_tensorboard


def main():
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="config/cifar10_resnet8.yml", help="Configuration file to use.")
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=Loader)

    roar_core.validate_configuration(cfg)

    dataset_name = cfg['data']['dataset']
    model_args = cfg['train_cls']['model']
    scheduler_args = cfg['train_cls']['scheduler']
    optimizer_args = cfg['train_cls']['optimizer']

    # Common Configuration
    dataset_args = cfg['data']

    train_data_args = dict(
        batch_size=cfg['train_cls']['batch_size'],
        shuffle=True,
        enable_augmentation=True,
    )

    assert train_data_args['enable_augmentation'], 'Augmentation of dataset should be enabled for training models.'

    val_data_args = dict(
        batch_size=train_data_args['batch_size'] * 4,
        shuffle=False,
        validate_step_size=1,
    )

    arguments = dict(
        dataset_args=dataset_args,
        train_data_args=train_data_args,
        val_data_args=val_data_args,
        model_args=model_args,
        loss_args=cfg['train_cls']['loss'],
        optimizer_args=optimizer_args,
        scheduler_args=scheduler_args,
        outdir=cfg['outdir'],
        nb_epochs=cfg['train_cls']['nb_epochs'],
        random_seed=random.randint(0, 1000)
    )

    train_and_evaluate_model(arguments)


def train_and_evaluate_model(arguments):
    """
    Main Pipeline for training and cross-validation.
    """

    """ Setup result directory and enable logging to file in it """
    logger.init(arguments['outdir'], filename_prefix='train_cls', log_level=logging.INFO)  # keep logs at root dir.
    outdir = os.path.join(arguments['outdir'], 'train_cls')
    os.makedirs(outdir, exist_ok=True)
    logger.info('Arguments:\n{}'.format(pformat(arguments)))

    """ Set random seed throughout python"""
    utils.set_random_seed(random_seed=arguments['random_seed'])

    """ Create tensorboard writer """
    tb_writer = initialize_tensorboard(outdir)

    """ Set device - cpu or gpu """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device - {device}')

    """ Load parameters for the Dataset """
    dataset = create_dataset(arguments['dataset_args'],
                             arguments['train_data_args'],
                             arguments['val_data_args'])

    """ Load Model with weights(if available) """
    model: torch.nn.Module = models_utils.get_model(
        arguments.get('model_args'), device, arguments['dataset_args']
    ).to(device)

    """ Create optimizer and scheduler """
    optimizer = optimizer_utils.create_optimizer(model.parameters(), arguments['optimizer_args'])
    lr_scheduler: _LRScheduler = optimizer_utils.create_scheduler(optimizer, arguments['scheduler_args'])

    """ Create loss function """
    logger.info(f"Loss weights {dataset.pos_neg_balance_weights()}")
    criterion = loss_utils.create_loss(arguments['loss_args'])

    """ Sample and View the inputs to model """
    dataset.debug()

    """ Pipeline - loop over the dataset multiple times """
    max_validation_acc, best_validation_model_path = 0, None
    batch_index = 0
    nb_epochs = 1 if is_debug_mode() else arguments['nb_epochs']
    for epoch in range(nb_epochs):
        """ Train the model """
        logger.info(f"Training, Epoch {epoch + 1}/{nb_epochs}")
        train_dataloader = dataset.train_dataloader
        model.train()
        start = time.time()
        total, correct = 0, 0
        epoch_loss = 0
        for i, data in enumerate(tqdm(train_dataloader)):
            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            tb_writer.save_scalar('batch_training_loss', loss.item(), batch_index)
            batch_index += 1
            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            optimizer.step()

        epoch_loss = epoch_loss / total
        logger.info(f"Epoch = {epoch}, Train_loss = {epoch_loss}, "
                    f"Time taken = {time.time() - start} seconds.")

        logger.info(f"Train_accuracy = {100 * correct / total}")
        tb_writer.save_scalar('training_loss', epoch_loss, epoch)
        tb_writer.save_scalar('training_acc', 100 * correct / total, epoch)

        """ Validate the model """
        val_data_args = arguments['val_data_args']
        if val_data_args['validate_step_size'] > 0 and \
                epoch % val_data_args['validate_step_size'] == 0:

            model.eval()
            validation_dataloader = dataset.validation_dataloader
            logger.info(f"Validation, Epoch {epoch + 1}/{arguments['nb_epochs']}")

            val_loss, val_accuracy = evaluate_single_class(device, model, validation_dataloader, criterion)
            logger.info(f'validation images: {dataset.val_dataset_size}, '
                        f'val_auc : {val_accuracy} %% '
                        f'val_loss: {val_loss}')
            tb_writer.save_scalar('validation_acc', val_accuracy, epoch)
            tb_writer.save_scalar('validation_loss', val_loss, epoch)

            """ Save Model """
            if val_accuracy > max_validation_acc:
                max_validation_acc = val_accuracy
                if best_validation_model_path is not None:
                    os.remove(best_validation_model_path)
                best_validation_model_path = os.path.join(outdir,
                                                          f'epoch_{epoch:04}-model-val_acc_{val_accuracy}.pth')
                torch.save(model.state_dict(), best_validation_model_path)
                logger.info(f'Model saved at: {best_validation_model_path}')

        if lr_scheduler:
            prev_lr = lr_scheduler.get_last_lr()
            lr_scheduler.step()
            if lr_scheduler.get_last_lr() != prev_lr:
                logger.warn(f'Updated LR from {prev_lr} to {lr_scheduler.get_lr()}')

    logger.info('Finished Training')
    logger.info(f'Max Validation accuracy is {max_validation_acc}')
    """ Create a symbolic link to the best model at a static path 'best_model.pth' """
    symlink_path = os.path.join(outdir, 'best_model.pth')
    if os.path.islink(symlink_path):
        os.unlink(symlink_path)
    os.symlink(best_validation_model_path.rsplit('/')[-1], symlink_path)
    logger.info(f'Best Model saved at: {best_validation_model_path}. and symlink to {symlink_path}')

    """ Evaluate model on test set """
    model.load_state_dict(torch.load(best_validation_model_path), strict=False)
    test_dataloader = dataset.test_dataloader
    test_loss, test_accuracy = evaluate_single_class(device, model, test_dataloader, criterion)
    logger.info(f'Accuracy of the network on the {dataset.test_dataset_size} test images: {test_accuracy} %%')
    return test_loss, test_accuracy


def evaluate_single_class(device, model, dataloader, criterion):
    correct, total_samples = 0, 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total_samples
    total_loss /= total_samples
    return total_loss, accuracy


if __name__ == '__main__':
    main()
