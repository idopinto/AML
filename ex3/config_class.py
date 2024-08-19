from dataclasses import dataclass
from pathlib import Path

import torch
import functools

# defining path for each directory
models_dir =  Path("./models")
models_dir.mkdir(parents=True, exist_ok=True)

collectors_dir = Path("./collectors")
collectors_dir.mkdir(parents=True, exist_ok=True)

data_dir = Path("./data")
data_dir.mkdir(parents=True, exist_ok=True)

plots_dir = Path("./plots")
plots_dir.mkdir(parents=True, exist_ok=True)

# setting seed for reproducability
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
print(f"Using {device} device")

# written with the help of ChatGPT
@dataclass
class Config:
    optimizer: type = torch.optim.Adam
    learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 1e-6
    lambda_: int = 25
    mu: int = 25
    nu: int = 1
    gamma: int = 1
    eps: float = 1e-4
    proj_dim: int = 512
    epochs: int = 30
    batch_size: int = 256
    D: int = 128

    def get_optimizer(self, params):
        # Fixate the optimizer parameters
        optimizer_func = functools.partial(
            self.optimizer,
            lr=self.learning_rate,
            betas=self.betas,
            weight_decay=self.weight_decay
        )
        return optimizer_func(params)

    def with_overrides(self, **overrides):
        # Create a new Config with overridden epochs or mu using partial
        return functools.partial(Config, **{**self.__dict__, **overrides})()