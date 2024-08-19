from safetensors import torch
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import random


class CustomCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.dataset = CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, labels = self.dataset[index]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2, labels

class OriginalCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.dataset = CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, labels = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, labels



class CIFAR10WithNeighbors(Dataset):
    def __init__(self, dataset, neighbors_indices):
        self.dataset = dataset
        self.neighbors_indices = neighbors_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, labels = self.dataset[idx]
        neighbor_idx = random.choice(self.neighbors_indices[idx, 1:])
        neighbor_image, _ = self.dataset[int(neighbor_idx)]
        return image, neighbor_image, labels

class CustomDatasetWithSingleLabel(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.label = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.label

class CIFAR10_MIXED_WITH_MNIST(Dataset):
    def __init__(self, root, transform=None, download=False):
        self.cifar10= CIFAR10(root=root, train=False, download=download, transform=None)
        self.mnist = MNIST(root=root, train=False, download=download, transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize if needed
        ]))
        self.transform = transform
        self.dataset = ConcatDataset([CustomDatasetWithSingleLabel(self.cifar10, 0), CustomDatasetWithSingleLabel(self.mnist,1)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
#
# class CIFAR10_MIXED_WITH_MNIST(Dataset):
#     def __init__(self, root, transform=None, download=False):
#         self.CIFAR_dataset = CIFAR10(root=root, train=False, download=download, transform=transforms.Compose([
#             transforms.ToTensor(),  # Convert CIFAR images to tensors
#             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize if needed
#         ]))
#         self.MNIST_dataset = MNIST(root=root, train=False, download=download, transform=transforms.Compose([
#             transforms.Grayscale(num_output_channels=3),
#             transforms.Resize((32, 32)),
#             transforms.ToTensor(),
#             # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize if needed
#         ]))
#         self.transform = transform
#
#         # Concatenate the targets
#         self.targets = [0] * len(self.CIFAR_dataset) + [1] * len(self.MNIST_dataset)
#
#     def __len__(self):
#         return len(self.targets)
#
#     def __getitem__(self, index):
#         if index < len(self.CIFAR_dataset):
#             img, _ = self.CIFAR_dataset[index]
#         else:
#             img, _ = self.MNIST_dataset[index - len(self.CIFAR_dataset)]
#
#         label = self.targets[index]
#         if self.transform:
#             img = self.transform(img)
#         return img, label
