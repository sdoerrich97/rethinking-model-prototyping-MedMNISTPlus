"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script evaluates a model on a specified dataset of the MedMNIST+ collection.
"""

# Import packages
import argparse
import yaml
import torch
import torch.nn as nn
import timm
import time
import medmnist
import random
import numpy as np
import torchvision.transforms as transforms

from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights
from medmnist import INFO
from sklearn.neighbors import NearestNeighbors

# Import custom modules
from utils import (calculate_passed_time, seed_worker, extract_embeddings, extract_embeddings_alexnet,
                   extract_embeddings_densenet, knn_majority_vote, get_ACC, get_AUC, get_ACC_kNN)


def evaluate(config: dict, train_loader: DataLoader, test_loader: DataLoader):
    """
    Evaluate a model on the specified dataset.

    :param config: Dictionary containing the parameters and hyperparameters.
    :param train_loader: DataLoader for the training set.
    :param test_loader: DataLoader for the test set.
    """

    # Start code
    start_time = time.time()

    # Load the trained model
    print("\tLoad the trained model ...")
    if config['architecture'] == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)

        if config['training_procedure'] == 'kNN':
            model.classifier = nn.Sequential(*[model.classifier[i] for i in range(6)])
        else:
            model.classifier[6] = nn.Linear(4096, config['num_classes'])

    else:
        model = timm.create_model(config['architecture'], pretrained=True, num_classes=config['num_classes'])

    if config['training_procedure'] == 'kNN':
        pass

    elif config['training_procedure'] == 'endToEnd' or config['training_procedure'] == 'linearProbing':
        checkpoint_file = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_{config['training_procedure']}_{config['architecture']}_s{config['seed']}_best.pth"
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint)

    else:
        raise ValueError("Training procedure not supported.")

    # Move the model to the available device
    model = model.to(config['device'])
    model.requires_grad_(False)
    model.eval()

    if config['training_procedure'] == 'kNN':
        print("\tCreate the support set ...")
        # Create the support set on the training data
        if config['architecture'] == 'alexnet':
            support_set = extract_embeddings_alexnet(model, config['device'], train_loader)

        elif config['architecture'] == 'densenet121':
            support_set = extract_embeddings_densenet(model, config['device'], train_loader)

        else:
            support_set = extract_embeddings(model, config['device'], train_loader)

        # Fit the NearestNeighbors model on the support set
        nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(support_set['embeddings'])

    # Define the output layer
    if config['task'] == "multi-label, binary-class":
        prediction = nn.Sigmoid()
    else:
        prediction = nn.Softmax(dim=1)

    # Run the Evaluation
    print(f"\tRun the evaluation ...")
    y_true, y_pred = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            # Map the data to the available device
            images, labels = images.to(config['device']), labels.to(torch.float32).to(config['device'])

            if config['training_procedure'] == 'kNN':
                if config['architecture'] == 'alexnet':
                    outputs = model(images)

                elif config['architecture'] == 'densenet121':
                    outputs = model.forward_features(images)
                    outputs = model.global_pool(outputs)

                else:
                    outputs = model.forward_features(images)
                    outputs = model.forward_head(outputs, pre_logits=True)

                outputs = outputs.reshape(outputs.shape[0], -1)
                outputs = outputs.detach().cpu().numpy()
                outputs = knn_majority_vote(nbrs, outputs, support_set['labels'], config['task'])
                outputs = outputs.to(config['device'])

            else:
                outputs = model(images)
                outputs = prediction(outputs)

            # Store the labels and predictions
            y_true = torch.cat((y_true, deepcopy(labels)), 0)
            y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)

        # Calculate the metrics
        if config['training_procedure'] == 'kNN':
            ACC = get_ACC_kNN(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
            AUC = 0.0  # AUC cannot be calculated for the kNN approach

        else:
            ACC = get_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
            AUC = get_AUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])

        # Print the loss values and send them to wandb
        print(f"\t\t\tACC: {ACC}")
        print(f"\t\t\tAUC: {AUC}")

    print(f"\tFinished evaluation.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time for evaluation: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--dataset", required=False, type=str, help="Which dataset to use.")
    parser.add_argument("--img_size", required=False, type=int, help="Which image size to use.")
    parser.add_argument("--training_procedure", required=False, type=str, help="Which training procedure to use.")
    parser.add_argument("--architecture", required=False, type=str, help="Which architecture to use.")
    parser.add_argument("--k", required=False, type=int, help="Number of nearest neighbors to use.")
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

    if args.k:
        config['k'] = args.k

    if args.seed:
        config['seed'] = args.seed

    # Seed the training and data loading so both become deterministic
    if config['architecture'] == 'alexnet':
        torch.backends.cudnn.benchmark = True  # Enable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = False  # Disable cudnn to be deterministic
        torch.use_deterministic_algorithms(False)  # Disable only deterministic algorithms

    else:
        torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in cudnn
        torch.backends.cudnn.deterministic = True  # Enable cudnn to be deterministic

        if config['architecture'] == 'samvit_base_patch16':
            torch.use_deterministic_algorithms(True, warn_only=True)  # Enable only deterministic algorithms

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
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0, padding_mode='constant') # Pad the image to 224x224
    ])

    # Create the datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])
    test_dataset = DataClass(split='test', transform=data_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    # Run the training
    evaluate(config, train_loader, test_loader)
