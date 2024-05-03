"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
Extract the latent features (embeddings) from all MedMNIST+ datasets before the classifier head for all specified
backbones.
"""

# Import packages
import argparse
import torch
import torch.nn as nn
import timm
import time
import medmnist
import numpy as np
import torchvision.transforms as transforms

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights
from medmnist import INFO

# Import custom modules
from utils import seed_worker


def extract_embeddings(model, device, dataloader, alexnet=False):
    """
    Extracts the embeddings from the model.

    :param model: Model.
    :param device: Device.
    :param dataloader: Dataloader.
    """
    embeddings_db, labels_db = [], []

    with torch.no_grad():
        for extracted in tqdm(dataloader):
            images, labels = extracted
            images = images.to(device)

            if alexnet:
                output = model(images)
            else:
                output = model.forward_features(images)
                output = model.forward_head(output, pre_logits=True)
                output = output.reshape(output.shape[0], -1)

            labels_db.append(deepcopy(labels.cpu().numpy()))
            embeddings_db.append(deepcopy(output.cpu().numpy()))

    data = {
        'embeddings': np.concatenate(embeddings_db, axis=0),
        'labels': np.concatenate(labels_db, axis=0),
    }

    return data


def main(data_path:str, output_path: str, batch_size: int = 256, device: str = 'cuda:0'):
    """
    Extracts the embeddings from the models.

    :param data_path: Path to the MedMNIST+ dataset.
    :param output_path: Path to the output folder.
    :param batch_size: Batch size.
    :param device: Device.
    """

    # Start code
    start_time = time.time()
    print("Run feature extraction ...")

    # Iterate over all MedMNIST+ datasets
    for dataset in ['bloodmnist', 'breastmnist', 'chestmnist', 'dermamnist', 'octmnist', 'organamnist', 'organcmnist',
                    'organsmnist', 'pathmnist', 'pneumoniamnist', 'retinamnist', 'tissuemnist']:

        print(f"\t... for {dataset}...")
        # Extract the dataset and its metadata
        info = INFO[dataset]
        task, in_channel, num_classes = info['task'], info['n_channels'], len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        # Iterate over all image sizes
        for img_size in [28, 64, 128, 224]:
            print(f"\t\t... for size {img_size}...")

            # Store the embeddings of each model for a specific dataset
            embeddings = {}

            # Create the padding depending on the image size
            total_padding = max(0, 224 - img_size)
            padding_left, padding_top = total_padding // 2, total_padding // 2
            padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top

            for architecture in ['vgg16', 'alexnet', 'resnet18', 'densenet121', 'efficientnet_b4','vit_base_patch16_224',
                                 'vit_base_patch16_clip_224', 'eva02_base_patch16_clip_224','vit_base_patch16_224.dino',
                                 'samvit_base_patch16']:
                print(f"\t\t\t ... for {architecture}...")

                # Create the model
                if architecture == 'alexnet':
                    model = alexnet(weights=AlexNet_Weights.DEFAULT)
                    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])  # Remove the classifier
                else:
                    model = timm.create_model(architecture, pretrained=True, num_classes=num_classes)

                # Freeze the model
                for param in model.parameters():
                    param.requires_grad = False

                # Move the model to the available device
                model = model.to(device)

                # Create the data transforms
                if architecture == 'alexnet':
                    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Use ImageNet statistics
                else:
                    m = timm.create_model(architecture, pretrained=True)
                    mean, std = m.default_cfg['mean'], m.default_cfg['std']

                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0, padding_mode='constant') # Pad the image to 224x224
                ])

                # Create the datasets
                train_dataset = DataClass(split='train', transform=data_transform, download=False, as_rgb=True, size=img_size, root=data_path)
                val_dataset = DataClass(split='val', transform=data_transform, download=False, as_rgb=True, size=img_size, root=data_path)
                test_dataset = DataClass(split='test', transform=data_transform, download=False, as_rgb=True, size=img_size, root=data_path)

                # Create the dataloaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

                # Store the embeddings of the current backbone architecture
                embeddings[architecture] = {}

                for set_name, dataloader in zip(["train", "val", "test"], [train_loader, val_loader, test_loader]):
                    print(f"\t\t\t\t ... for the {set_name} set...")

                    # Extract the embeddings of the current backbone architecture
                    if architecture == 'alexnet':
                        data = extract_embeddings(model, device, dataloader, alexnet=True)
                    else:
                        data = extract_embeddings(model, device, dataloader)

                    # Store the embeddings
                    embeddings[architecture][set_name] = data

            # Store the embedding
            # Create a unique filename
            if img_size == 28:
                filename = Path(output_path) / f"{dataset}_embeddings.npz"
            else:
                filename = Path(output_path) / f"{dataset}_{img_size}_embeddings.npz"

            # Save the embeddings and labels into a npz file
            np.savez(filename, embeddings)


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str, help="Path to the MedMNIST+ dataset.")
    parser.add_argument("--output_path", required=True, type=str, help="Path to the output folder.")
    parser.add_argument("--batch_size", required=False, type=int, default=256, help="Which dataset to use.")
    parser.add_argument("--device", required=False, type=str, default='cuda:0', help="Which image size to use.")

    args = parser.parse_args()
    main(data_path=args.data_path, output_path=args.output_path, batch_size=args.batch_size, device=args.device)
