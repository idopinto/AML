import torch
from torch import nn
from torch.distributions import MultivariateNormal


#######################################################################
############################# Models ##################################
#######################################################################
class AffineCouplingLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(AffineCouplingLayer, self).__init__()
        self.scale_linear_layers = nn.Sequential(
            nn.Linear(in_features // 2, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, in_features // 2, bias=bias),
        )
        self.shift_linear_layers = nn.Sequential(
            nn.Linear(in_features // 2, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, in_features // 2, bias=bias),
        )

    def forward(self, z):
        zl, zr = z.chunk(2, dim=1)
        log_s, b = self.scale_linear_layers(zl), self.shift_linear_layers(zl)
        yl, yr = zl, log_s.exp() * zr + b
        log_det_jacboian = log_s.sum(dim=1)
        return torch.cat([yl, yr], dim=1), log_det_jacboian

    def inverse(self, y):
        yl, yr = y.chunk(2, dim=1)
        log_s, b = self.scale_linear_layers(yl), self.shift_linear_layers(yl)
        zl, zr = yl, (yr - b) / torch.exp(log_s)
        log_det_jacobian = -log_s.sum(dim=1)
        return torch.cat([zl, zr], dim=1), log_det_jacobian

    # def compute_log_inverse_jacobian_det(self, y):
    #     yl, yr = y.chunk(2, dim=1)
    #     log_s = self.scale_linear_layers(yl)
    #     return -log_s.sum(dim=1)


class PermutationLayer(nn.Module):
    def __init__(self, input_size):
        super(PermutationLayer, self).__init__()
        self.input_size = input_size
        self.permutation_indices = torch.randperm(self.input_size)
        self.inverse_permutation_indices = torch.argsort(self.permutation_indices)

    def forward(self, x):
        self.permutation_indices = self.permutation_indices.to(x.device)
        return x[:, self.permutation_indices], torch.zeros(x.size(0), device=x.device)

    def inverse(self, y):
        self.inverse_permutation_indices = self.inverse_permutation_indices.to(y.device)
        return y[:, self.inverse_permutation_indices], torch.zeros(y.size(0), device=y.device)


class NormalizingFlowModel(nn.Module):
    def __init__(self, n_layers, in_features, out_features, bias=True):
        super(NormalizingFlowModel, self).__init__()
        self.interleaving_affine_coupling_layers = nn.ModuleList()
        for i in range(n_layers):
            self.interleaving_affine_coupling_layers.append(
                AffineCouplingLayer(in_features, out_features, bias=bias)
            )
            self.interleaving_affine_coupling_layers.append(PermutationLayer(in_features))
        self.model_name = "nf"

    def forward(self, z):
        for i, layer in enumerate(self.interleaving_affine_coupling_layers):
            z, log_det = layer(z)
        return z

    def inverse(self, y):
        log_inv_jacobian_det = torch.zeros(y.size(0), device=y.device)
        for i, layer in enumerate(reversed(self.interleaving_affine_coupling_layers)):
            y, log_det = layer.inverse(y)
            log_inv_jacobian_det += log_det
        return y, log_inv_jacobian_det

    def log_probability(self, y):
        z, log_inv_jacobian_det = self.inverse(y)
        Pz = MultivariateNormal(torch.zeros(z.shape[1], device=z.device),
                                torch.eye(z.shape[1], device=z.device))
        prior_log_probability = Pz.log_prob(z)
        return prior_log_probability + log_inv_jacobian_det, prior_log_probability, log_inv_jacobian_det


class UnconditionalFlowMatchingModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(UnconditionalFlowMatchingModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features + 1, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, out_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(out_features, in_features, bias=bias),
        )
        self.model_name = "fm"

    def forward(self, y, t):
        y_t = torch.cat((y, t), dim=1)
        return self.fc_layers(y_t)


class ConditionalFlowMatchingModel(UnconditionalFlowMatchingModel):
    def __init__(self, in_features, out_features, num_classes, embedding_dim, bias=True):
        super(ConditionalFlowMatchingModel, self).__init__(in_features, out_features, bias)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.fc_layers[0] = nn.Linear(in_features=in_features + embedding_dim + 1, out_features=out_features)
        self.model_name = "cfm"
    def forward(self, y, t, label=None):
        label_embed = self.embedding(label)
        y_t = torch.cat((y, t, label_embed), dim=1)
        return self.fc_layers(y_t)


###########################################################################
############################# Criterions ##################################
###########################################################################
class FlowMatchingCriterion(nn.Module):
    def __init__(self):
        super(FlowMatchingCriterion, self).__init__()

    def forward(self, v_hat_t, y_0, y_1):
        return (torch.linalg.norm(v_hat_t - (y_1 - y_0)) ** 2).mean()


class NormalizingFlowCriterion(nn.Module):
    def __init__(self):
        super(NormalizingFlowCriterion, self).__init__()

    def forward(self, prior_log_probability, log_inv_jacobian_det):
        return (-prior_log_probability - log_inv_jacobian_det).mean()
