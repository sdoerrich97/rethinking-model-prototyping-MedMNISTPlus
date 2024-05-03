"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Main script to train and evaluate a model on the specified dataset of the MedMNIST+ collection.
"""

# Import packages
import argparse
import yaml
import torch
import timm
import time
import medmnist
import random
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from medmnist import INFO

# Import custom modules
from train import train
from evaluate import evaluate
from utils import calculate_passed_time, seed_worker


def main(config: dict):
    """
    Main function to train and evaluate a model on the specified dataset.

    :param config: Dictionary containing the parameters and hyperparameters.
    """

    # Start code
    start_time = time.time()
    print("\tRun Details:")
    print("\t\tDataset: {}".format(config['dataset']))
    print("\t\tImage size: {}".format(config['img_size']))
    print("\t\tTraining procedure: {}".format(config['training_procedure']))
    print("\t\tArchitecture: {}".format(config['architecture']))
    print("\t\tSeed: {}".format(config['seed']))

    # Seed the training and data loading so both become deterministic
    print("\tSeed:")
    if config['architecture'] == 'alexnet':
        torch.backends.cudnn.benchmark = True  # Enable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = False  # Disable cudnn to be deterministic
        torch.use_deterministic_algorithms(False)  # Disable only deterministic algorithms

    else:
        torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = True  # Enable cudnn to be deterministic

        if config['architecture'] == 'samvit_base_patch16':
            torch.use_deterministic_algorithms(True, warn_only=True)

        else:
            torch.use_deterministic_algorithms(True)  # Enable only deterministic algorithms

    torch.manual_seed(config['seed'])  # Seed the pytorch RNG for all devices (both CPU and CUDA)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    g = torch.Generator()
    g.manual_seed(config['seed'])

    # Extract the dataset and its metadata
    info = INFO[config['dataset']]
    config['task'], config['in_channel'], config['num_classes'] = info['task'], info['n_channels'], len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    # Create the data transforms and normalize with imagenet statistics
    if config['architecture'] == 'alexnet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Use ImageNet statistics
    else:
        m = timm.create_model(config['architecture'], pretrained=True)
        mean, std = m.default_cfg['mean'], m.default_cfg['std']

    total_padding = max(0, 224 - config['img_size'])
    padding_left, padding_top = total_padding // 2, total_padding // 2
    padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0, padding_mode='constant')  # Pad the image to 224x224
    ])

    # Create the datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])
    val_dataset = DataClass(split='val', transform=data_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])
    test_dataset = DataClass(split='test', transform=data_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    train_loader_at_eval = DataLoader(train_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    # Run the training
    if config['training_procedure'] == 'endToEnd' or config['training_procedure'] == 'linearProbing':
        train(config, train_loader, val_loader)
    elif config['training_procedure'] == 'kNN':
        pass
    else:
        raise ValueError("The specified training procedure is not supported.")

    # Run the evaluation
    evaluate(config, train_loader_at_eval, test_loader)

    print(f"\tFinished current run.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--dataset", required=False, type=str, help="Which dataset to use.")
    parser.add_argument("--img_size", required=False, type=int, help="Which image size to use.")
    parser.add_argument("--training_procedure", required=False, type=str, help="Which training procedure to use.")
    parser.add_argument("--architecture", required=False, type=str, help="Which architecture to use.")
    parser.add_argument("--seed", required=False, type=int, help="Which seed was used during training.")

    args = parser.parse_args()
    config_file = args.config_file

    # Load the parameters and hyperparameters of the configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Adapt to the command line arguments
    if args.dataset:
        config['dataset'] = args.dataset

    if args.img_size:
        config['img_size'] = args.img_size

    if args.training_procedure:
        config['training_procedure'] = args.training_procedure

    if args.architecture:
        config['architecture'] = args.architecture

    # If a seed is specified, overwrite the seed in the config file
    if args.seed:
        config['seed'] = args.seed

    # Run
    main(config)
