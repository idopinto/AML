import torch
from torch import nn
class VICRegCriterion(nn.Module):
  def __init__(self,lam, mu, nu, gamma, eps):
    super().__init__()
    self.lam = lam
    self.mu = mu
    self.nu = nu
    self.gamma = gamma
    self.eps = eps
    self.inv_loss = InvarianceCriterion()
    self.var_loss = VarianceCriterion(self.gamma, self.eps)
    self.cov_loss = CovarianceCriterion()

  def forward(self, Z, Z_tag):
     inv_loss = self.inv_loss(Z, Z_tag)
     var_loss = self.var_loss(Z) + self.var_loss(Z_tag)
     cov_loss = self.cov_loss(Z) + self.cov_loss(Z_tag)
     loss_components = [inv_loss, var_loss, cov_loss]
     loss = (self.lam * inv_loss) + (self.mu * var_loss) + (self.nu * cov_loss)
     return loss, loss_components

class InvarianceCriterion(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, Z, Z_tag):
    mse = torch.nn.MSELoss()
    inv_loss = mse(Z, Z_tag)
    return inv_loss

class VarianceCriterion(nn.Module):
  def __init__(self, gamma, eps):
    super().__init__()
    self.gamma = gamma
    self.eps = eps

  def forward(self, Z):
    sigma = torch.sqrt(torch.var(Z, dim=0) + self.eps)
    var_loss = torch.mean(nn.ReLU()(self.gamma - sigma))
    return var_loss

class CovarianceCriterion(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, Z):
    B, d = Z.shape
    mean_Z = torch.mean(Z, dim=0)
    Z_centered = Z - mean_Z
    cov_Z = torch.mm(Z_centered.T, Z_centered) / (B - 1)
    cov_loss = torch.sum(torch.square(cov_Z * (1 - torch.eye(d, device=Z.device)))) / d
    return cov_loss