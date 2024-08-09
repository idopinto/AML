import torch
import torchvision
import torchvision.transforms as transforms
import models, engine, utils
from torch import nn
from matplotlib import pyplot as plt
import numpy as np
def main():
    config = {
        "optimizer": torch.optim.Adam,
        "learning_rate": 3e-4,
        "betas": (0.9, 0.999),
        "weight_decay": 1e-6,
        "lambda": 25,
        "mu": 25,
        "nu": 1,
        "proj_dim": 512,
        "train_size": 250000,
        "validation_size": 50000,
        "epochs": 10,
        "batch_size": 256,
        "seed": 42,
        "embedding_dim": 10
    }

    seed = config["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    torch.manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Show images
    imshow(torchvision.utils.make_grid(images))

if __name__ == '__main__':
    main()
