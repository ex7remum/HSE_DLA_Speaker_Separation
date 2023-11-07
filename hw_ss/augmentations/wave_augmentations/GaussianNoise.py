import torch
from torch import Tensor
import random

from hw_ss.augmentations.base import AugmentationBase


class GaussianNoise(AugmentationBase):
    def __init__(self, p, std, *args, **kwargs):
        self.p = p
        self.distribution = torch.distributions.Normal(0, std)

    def __call__(self, data: Tensor):
        if random.random() <= self.p:
            out = data + self.distribution.sample(data.size())
        else:
            out = data
        return out
