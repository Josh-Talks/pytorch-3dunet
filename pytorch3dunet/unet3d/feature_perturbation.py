import torch
from torch import nn
import numpy as np
from torch.distributions.uniform import Uniform

class DropOutPerturbation(nn.Module):
    def __init__(self, drop_rate=0.3, spatial_dropout=True, **kwargs):
        super(DropOutPerturbation, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.dropout(x)
        return x
    

class FeatureDropPerturbation(nn.Module):
    def __init__(self, th_lower=0.7, th_upper=0.9, **kwargs):
        super(FeatureDropPerturbation, self).__init__()
        self.th_lower = th_lower
        self.th_upper = th_upper

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(self.th_lower, self.th_upper)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.feature_dropout(x)
        return x
    

class FeatureNoisePerturbation(nn.Module):
    def __init__(self, uniform_range=0.3, **kwargs):
        super(FeatureNoisePerturbation, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x