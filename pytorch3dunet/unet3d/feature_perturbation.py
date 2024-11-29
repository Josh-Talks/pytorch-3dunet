import torch
from torch import nn
import numpy as np
from torch.distributions.uniform import Uniform


class DropOutPerturbation(nn.Module):
    def __init__(self, drop_rate=0.3, spatial_dropout=True, random_seed=1, **kwargs):
        super(DropOutPerturbation, self).__init__()
        # self.dropout = (
        #    nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        # )
        self.dropout = Dropout2d(p=drop_rate) if spatial_dropout else Dropout(drop_rate)

    def forward(self, x):
        x = self.dropout(x)
        return x


class FeatureDropPerturbation(nn.Module):
    def __init__(self, th_lower=0.7, th_upper=0.9, random_seed=42, **kwargs):
        super(FeatureDropPerturbation, self).__init__()
        self.th_lower = th_lower
        self.th_upper = th_upper
        self.rng = np.random.default_rng(random_seed)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * self.rng.uniform(self.th_lower, self.th_upper)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.feature_dropout(x)
        return x


class FeatureNoisePerturbation(nn.Module):
    def __init__(self, uniform_range=0.3, random_seed=42, **kwargs):
        super(FeatureNoisePerturbation, self).__init__()
        # self.uni_dist = Uniform(-uniform_range, uniform_range)
        self.uni_range = uniform_range
        self.rng = torch.Generator().manual_seed(random_seed)

    def feature_based_noise(self, x):
        # noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)

        noise_vector = (
            (
                (-self.uni_range - self.uni_range)
                * torch.rand(
                    x.shape[1:],
                    generator=self.rng,
                )
                + (-self.uni_range)
            )
            .to(x.device)
            .unsqueeze(0)
        )
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


class Dropout(nn.Module):
    def __init__(self, p: float = 0.5, random_seed: int = 42):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.rng = torch.Generator().manual_seed(random_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor. shape (N, C, H, W)

        Returns:
            (torch.Tensor): Output tensor. shape (N, C, H, W)
        """
        mask = (
            torch.rand(
                x.size(), generator=self.rng
            )
            > self.p
        ).float().to(x.device)
        return mask * x * (1.0 / (1 - self.p))


class Dropout2d(nn.Module):
    def __init__(self, p: float = 0.5, random_seed: int = 42):
        super(Dropout2d, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p
        self.rng = torch.Generator().manual_seed(random_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor. shape (N, C, H, W)

        Returns:
            (torch.Tensor): Output tensor. shape (N, C, H, W)
        """
        mask = (
            torch.rand(
                (x.size(0), x.size(1), 1, 1),
                generator=self.rng,
            )
            > self.p
        ).float().to(x.device)
        return mask * x * (1.0 / (1 - self.p))
