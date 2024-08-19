import torch
import torch.nn as nn
from torchvision.models import resnet18


# f - encoder function
class Encoder(nn.Module):
    def __init__(self, D=128, device='cuda'):
        super(Encoder, self).__init__()
        self.resnet = resnet18(pretrained=False).to(device)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):
    def __init__(self, D, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   )

    def forward(self, x):
        return self.model(x)

class VICRegModel(nn.Module):
    def __init__(self, D=128, proj_dim=512, device='cpu'):
        super(VICRegModel, self).__init__()
        self.encoder = Encoder(D, device).to(device)
        self.projector = Projector(D, proj_dim).to(device)
        self.name = "vicreg"

    def forward(self, x):
        y = self.encoder(x)
        z = self.projector(y)
        return y, z


class VICRegLinearProbing(nn.Module):
  def __init__(self, vic_reg_model, D, num_classes):
    super().__init__()
    self.encoder = vic_reg_model.encoder
    self.fc = nn.Linear(in_features= D, out_features=num_classes)
    self.name = "vicreg_lp"
  def forward(self, x):
    y = self.encoder(x)
    z = self.fc(y)
    return z