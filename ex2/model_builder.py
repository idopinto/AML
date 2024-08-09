import torch
from torch import nn
from torch.distributions import MultivariateNormal


#######################################################################
############################# Models ##################################
#######################################################################

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer used in normalizing flows. It splits the input into two parts,
    and applies an affine transformation to one part based on the other.

    :param in_features: Number of input features.
    :param out_features: Number of output features in the intermediate layers.
    :param bias: Whether to include a bias term in the linear layers.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(AffineCouplingLayer, self).__init__()
        # Define the scale transformation network
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
        # Define the shift transformation network
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
        """
        Forward pass for the affine coupling layer.

        :param z: Input tensor.
        :return: Transformed output and the log-determinant of the Jacobian.
        """
        zl, zr = z.chunk(2, dim=1)  # Split the input into two halves
        log_s, b = self.scale_linear_layers(zl), self.shift_linear_layers(zl)  # Compute scale and shift
        yl, yr = zl, log_s.exp() * zr + b  # Apply affine transformation
        log_det_jacboian = log_s.sum(dim=1)  # Compute the log-determinant of the Jacobian
        return torch.cat([yl, yr], dim=1), log_det_jacboian

    def inverse(self, y):
        """
        Inverse pass for the affine coupling layer.

        :param y: Input tensor.
        :return: Inverse transformed output and the negative log-determinant of the Jacobian.
        """
        yl, yr = y.chunk(2, dim=1)  # Split the input into two halves
        log_s, b = self.scale_linear_layers(yl), self.shift_linear_layers(yl)  # Compute scale and shift
        zl, zr = yl, (yr - b) / torch.exp(log_s)  # Apply inverse affine transformation
        log_det_jacobian = -log_s.sum(dim=1)  # Compute the negative log-determinant of the Jacobian
        return torch.cat([zl, zr], dim=1), log_det_jacobian


class PermutationLayer(nn.Module):
    """
    Permutation layer used in normalizing flows. This layer randomly permutes the features of the input.

    :param input_size: Number of input features.
    """

    def __init__(self, input_size):
        super(PermutationLayer, self).__init__()
        self.input_size = input_size
        self.permutation_indices = torch.randperm(self.input_size)  # Generate random permutation indices
        self.inverse_permutation_indices = torch.argsort(
            self.permutation_indices)  # Compute inverse permutation indices

    def forward(self, x):
        """
        Forward pass for the permutation layer.

        :param x: Input tensor.
        :return: Permuted tensor and a tensor of zeros for log-determinant of the Jacobian.
        """
        self.permutation_indices = self.permutation_indices.to(x.device)
        return x[:, self.permutation_indices], torch.zeros(x.size(0), device=x.device)

    def inverse(self, y):
        """
        Inverse pass for the permutation layer.

        :param y: Input tensor.
        :return: Inverse permuted tensor and a tensor of zeros for log-determinant of the Jacobian.
        """
        self.inverse_permutation_indices = self.inverse_permutation_indices.to(y.device)
        return y[:, self.inverse_permutation_indices], torch.zeros(y.size(0), device=y.device)


class NormalizingFlowModel(nn.Module):
    """
    Normalizing flow model consisting of interleaved affine coupling and permutation layers.

    :param n_layers: Number of affine coupling layers.
    :param in_features: Number of input features.
    :param out_features: Number of output features in the intermediate layers.
    :param bias: Whether to include a bias term in the linear layers.
    """

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
        """
        Forward pass for the normalizing flow model.

        :param z: Input tensor.
        :return: Transformed output.
        """
        for i, layer in enumerate(self.interleaving_affine_coupling_layers):
            z, log_det = layer(z)
        return z

    def inverse(self, y):
        """
        Inverse pass for the normalizing flow model.

        :param y: Input tensor.
        :return: Inverse transformed output and the log-determinant of the Jacobian.
        """
        log_inv_jacobian_det = torch.zeros(y.size(0), device=y.device)
        for i, layer in enumerate(reversed(self.interleaving_affine_coupling_layers)):
            y, log_det = layer.inverse(y)
            log_inv_jacobian_det += log_det
        return y, log_inv_jacobian_det

    def log_probability(self, y):
        """
        Compute the log-probability of the input under the normalizing flow model.

        :param y: Input tensor.
        :return: Log-probability, prior log-probability, and log-determinant of the Jacobian.
        """
        z, log_inv_jacobian_det = self.inverse(y)
        Pz = MultivariateNormal(torch.zeros(z.shape[1], device=z.device),
                                torch.eye(z.shape[1], device=z.device))
        prior_log_probability = Pz.log_prob(z)
        return prior_log_probability + log_inv_jacobian_det, prior_log_probability, log_inv_jacobian_det


class UnconditionalFlowMatchingModel(nn.Module):
    """
    Unconditional flow matching model that learns to map between distributions.

    :param in_features: Number of input features.
    :param out_features: Number of output features in the intermediate layers.
    :param bias: Whether to include a bias term in the linear layers.
    """

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
        """
        Forward pass for the flow matching model.

        :param y: Input tensor.
        :param t: Time variable.
        :return: Transformed output.
        """
        y_t = torch.cat((y, t), dim=1)
        return self.fc_layers(y_t)


class ConditionalFlowMatchingModel(UnconditionalFlowMatchingModel):
    """
    Conditional flow matching model that includes class embeddings.

    :param in_features: Number of input features.
    :param out_features: Number of output features in the intermediate layers.
    :param num_classes: Number of classes for the embedding.
    :param embedding_dim: Dimension of the class embedding.
    :param bias: Whether to include a bias term in the linear layers.
    """

    def __init__(self, in_features, out_features, num_classes, embedding_dim, bias=True):
        super(ConditionalFlowMatchingModel, self).__init__(in_features, out_features, bias)
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.fc_layers[0] = nn.Linear(in_features=in_features + embedding_dim + 1, out_features=out_features)
        self.model_name = "cfm"

    def forward(self, y, t, label=None):
        """
        Forward pass for the conditional flow matching model.

        :param y: Input tensor.
        :param t: Time variable.
        :param label: Class label for the embedding.
        :return: Transformed output.
        """
        label_embed = self.embedding(label)
        y_t = torch.cat((y, t, label_embed), dim=1)
        return self.fc_layers(y_t)


###########################################################################
############################# Criterions ##################################
###########################################################################

class FlowMatchingCriterion(nn.Module):
    """
    Criterion for flow matching models, comparing the predicted velocity with the difference between
    the input and output.

    :return: Mean squared error loss.
    """

    def __init__(self):
        super(FlowMatchingCriterion, self).__init__()

    def forward(self, v_hat_t, y_0, y_1):
        """
        Forward pass for the criterion.

        :param v_hat_t: Predicted velocity.
        :param y_0: Initial state.
        :param y_1: Final state.
        :return: Mean squared error loss.
        """
        return (torch.linalg.norm(v_hat_t - (y_1 - y_0)) ** 2).mean()


class NormalizingFlowCriterion(nn.Module):
    """
    Criterion for normalizing flow models, combining the prior log-probability and
    the log-determinant of the Jacobian.

    :return: Negative log-likelihood loss.
    """

    def __init__(self):
        super(NormalizingFlowCriterion, self).__init__()

    def forward(self, prior_log_probability, log_inv_jacobian_det):
        """
        Forward pass for the criterion.

        :param prior_log_probability: Log-probability of the prior distribution.
        :param log_inv_jacobian_det: Log-determinant of the Jacobian.
        :return: Negative log-likelihood loss.
        """
        return (-prior_log_probability - log_inv_jacobian_det).mean()
