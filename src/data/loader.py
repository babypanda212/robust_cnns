import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms

batch_size = 1024 # batch size has to be < 2**16, should be <= 2**13 for T4
debug = True

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]),
    ])

""" Load data """
data_train = CIFAR10(root="datasets", train=True, download=True, transform=transform)
data_test = CIFAR10(root="datasets", train=False, download=True, transform=transform)
data_test, data_val = torch.utils.data.random_split(data_test, [0.1, 0.9])

dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(data_test, batch_size=len(data_test), shuffle=False) # create test dataloader with a single batch
dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)

num_classes = len(data_train.classes)

# mean = data_train.data.mean(axis=(0,1,2)) / 255 # [0.49139968, 0.48215841, 0.44653091]
# std = data_train.data.std(axis=(0,1,2)) / 255 # [0.24703223, 0.24348513, 0.26158784]

data_train_subset = Subset(data_train, list(range(2*batch_size)))
data_val_subset = Subset(data_val, list(range(10)))
data_test_subset = Subset(data_test, list(range(100)))

dataloader_train_subset = DataLoader(data_train_subset, batch_size=batch_size, shuffle=True)
dataloader_val_subset = DataLoader(data_val_subset, batch_size=len(data_val_subset), shuffle=False)
dataloader_test_subset = DataLoader(data_test_subset, batch_size=len(data_test_subset), shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")