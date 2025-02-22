# src/data/loader.py
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

def create_cifar10_loaders(batch_size=1024, debug=False):
    """
    Creates CIFAR-10 data loaders with train, validation, and test sets.

    Args:
        batch_size (int): Batch size for the data loaders.
        debug (bool): If True, use a smaller subset of the data for debugging.

    Returns:
        tuple: (dataloader_train, dataloader_val, dataloader_test),
               or (dataloader_train_subset, dataloader_val_subset, dataloader_test_subset) if debug is True
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
    ])

    # Load data
    data_train = CIFAR10(root="datasets", train=True, download=True, transform=transform)
    data_test = CIFAR10(root="datasets", train=False, download=True, transform=transform)
    data_test, data_val = torch.utils.data.random_split(data_test, [0.1, 0.9]) # Split the test set


    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=len(data_test), shuffle=False) # create test dataloader with a single batch
    dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)

    if debug:
        data_train_subset = Subset(data_train, list(range(2*batch_size)))
        data_val_subset = Subset(data_val, list(range(10)))
        data_test_subset = Subset(data_test, list(range(100)))

        dataloader_train_subset = DataLoader(data_train_subset, batch_size=batch_size, shuffle=True)
        dataloader_val_subset = DataLoader(data_val_subset, batch_size=len(data_val_subset), shuffle=False)
        dataloader_test_subset = DataLoader(data_test_subset, batch_size=len(data_test_subset), shuffle=False)

        return dataloader_train_subset, dataloader_val_subset, dataloader_test_subset

    return dataloader_train, dataloader_val, dataloader_test

def get_num_classes(dataloader_train):
    """Returns the number of classes in the training dataset."""
    return len(dataloader_train.dataset.classes)

def get_device():
    """Returns the device (cuda or cpu) based on availability."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
