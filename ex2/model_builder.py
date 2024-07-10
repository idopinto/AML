import torch
from torch import nn


class AffineCouplingLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(AffineCouplingLayer, self).__init__()
        pass

    def forward(self, z):
        pass

    def inverse(self, y):
        pass

    def compute_log_inverse_jacobian_det(self):
        pass


class PermutationLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(PermutationLayer, self).__init__()
        pass

    def forward(self, x):
        pass

    def inverse(self, y):
        pass


class NormalizingFlowModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(NormalizingFlowModel, self).__init__()
        pass

    def forward(self, x):
        pass

    def inverse(self, y):
        pass

    def compute_log_inverse_jacobian_det(self):
        pass
    
