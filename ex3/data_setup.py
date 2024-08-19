from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, Dataset
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

# class CIFAR10WithNeighbors(Dataset):
#     def __init__(self, root , train=True, transform=None, download=False, neighbors_indices=None):
#         self.dataset = CIFAR10(root=root, train=train, download=download)
#         self.neighbors_indices = neighbors_indices
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         image, label = self.dataset[idx]
#         neighbor_idx = random.choice(self.neighbors_indices[idx, 1:])
#         nn_image, _ = self.dataset[neighbor_idx]
#         return image, nn_image, label