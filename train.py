"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script trains a model on a specified dataset of the MedMNIST+ collection and saves the best performing model.
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

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from torchvision.models import alexnet, AlexNet_Weights
from medmnist import INFO

# Import custom modules
from utils import calculate_passed_time, seed_worker, get_ACC, get_AUC


def train(config: dict, train_loader: DataLoader, val_loader: DataLoader):
    """
    Train a model on the specified dataset and save the best performing model.

    :param config: Dictionary containing the parameters and hyperparameters.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    """

    # Start code
    start_time = time.time()
    print("\tStart training ...")

    # Create the model
    print("\tCreate the model ...")
    if config['architecture'] == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, config['num_classes'])
    else:
        model = timm.create_model(config['architecture'], pretrained=True, num_classes=config['num_classes'])

    # Initialize the model for the given training procedure
    print("\tInitialize the model for the given training procedure ...")
    if config['training_procedure'] == 'endToEnd':
        pass

    elif config['training_procedure'] == 'linearProbing':
        # Set only the last layer to trainable
        for param in model.parameters():
            param.requires_grad = False

        if config['architecture'] == 'alexnet':
            for param in model.classifier[6].parameters():
                param.requires_grad = True
        else:
            for param in model.get_classifier().parameters():
                param.requires_grad = True

    elif config['training_procedure'] == 'kNN':
        # Freeze the model
        for param in model.parameters():
            param.requires_grad = False

    else:
        raise ValueError("Training procedure not supported.")

    # Move the model to the available device
    model = model.to(config['device'])

    # Create the optimizer and the learning rate scheduler
    print("\tCreate the optimizer and the learning rate scheduler ...")
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineLRScheduler(optimizer, t_initial=config['epochs'], cycle_limit=1, t_in_epochs=True)

    # Define the loss function
    print("\tDefine the loss function ...")
    if config['task'] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss().to(config['device'])
        prediction = nn.Sigmoid()
    else:
        criterion = nn.CrossEntropyLoss().to(config['device'])
        prediction = nn.Softmax(dim=1)

    # Create variables to store the best performing model
    print("\tInitialize helper variables ...")
    best_loss, best_epoch = np.inf, 0
    best_model = deepcopy(model)
    epochs_no_improve = 0  # Counter for epochs without improvement
    n_epochs_stop = 5  # Number of epochs to wait before stopping

    # Training loop
    print(f"\tRun the training for {config['epochs']} epochs ...")
    for epoch in range(config['epochs']):
        start_time_epoch = time.time()  # Stop the time
        print(f"\t\tEpoch {epoch} of {config['epochs']}:")

        # Training
        print(f"\t\t\t Train:")
        model.train()
        train_loss = 0

        for images, labels in tqdm(train_loader):
            # Map the data to the available device
            images = images.to(config['device'])

            if config['task'] == 'multi-label, binary-class':
                labels = labels.to(torch.float32).to(config['device'])
            else:
                labels = torch.squeeze(labels, 1).long().to(config['device'])

            # Run the forward pass
            outputs = model(images)

            # Compute the loss and perform backpropagation
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

        # Update the learning rate
        scheduler.step(epoch=epoch)

        # Evaluation
        print(f"\t\t\t Evaluate:")
        model.eval()
        val_loss = 0
        y_true, y_pred = torch.tensor([]).to(config['device']), torch.tensor([]).to(config['device'])

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                # Map the data to the available device
                images = images.to(config['device'])
                outputs = model(images)

                # Run the forward pass
                if config['task'] == 'multi-label, binary-class':
                    labels = labels.to(torch.float32).to(config['device'])
                    loss = criterion(outputs, labels)
                    outputs = prediction(outputs).to(config['device'])

                else:
                    labels = torch.squeeze(labels, 1).long().to(config['device'])
                    loss = criterion(outputs, labels)
                    outputs = prediction(outputs).to(config['device'])
                    labels = labels.float().resize_(len(labels), 1)

                # Store the current loss
                val_loss += loss.item()

                # Store the labels and predictions
                y_true = torch.cat((y_true, deepcopy(labels)), 0)
                y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)

        # Calculate the metrics
        ACC = get_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
        AUC = get_AUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])

        # Print the loss values and send them to wandb
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"\t\t\tTrain Loss: {train_loss}")
        print(f"\t\t\tVal Loss: {val_loss}")
        print(f"\t\t\tACC: {ACC}")
        print(f"\t\t\tAUC: {AUC}")

        # Store the current best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1  # Increment the counter

        print(f"\t\t\tCurrent best Val Loss: {best_loss}")
        print(f"\t\t\tCurrent best Epoch: {best_epoch}")

        # Check for early stopping
        if epochs_no_improve == n_epochs_stop:
            print("\tEarly stopping!")
            break

        # Stop the time for the epoch
        end_time_epoch = time.time()
        hours_epoch, minutes_epoch, seconds_epoch = calculate_passed_time(start_time_epoch, end_time_epoch)
        print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch, seconds_epoch))

    print(f"\tSave the trained model ...")
    Path(config['output_path']).mkdir(parents=True, exist_ok=True)
    save_name = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_{config['training_procedure']}_{config['architecture']}_s{config['seed']}"
    torch.save(model.state_dict(), f"{save_name}_final.pth")
    torch.save(best_model.state_dict(), f"{save_name}_best.pth")

    print(f"\tFinished training.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time for training: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

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
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0, padding_mode='constant')  # Pad the image to 224x224
    ])

    # Create the datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])
    val_dataset = DataClass(split='val', transform=data_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])

    # Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    # Run the training
    train(config, train_loader, val_loader)
