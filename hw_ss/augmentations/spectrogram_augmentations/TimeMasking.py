import torchaudio
from torch import Tensor

from hw_ss.augmentations.base import AugmentationBase
from hw_ss.augmentations.random_apply import RandomApply


class TimeMasking(AugmentationBase):
    def __init__(self, time_mask, p):
        self._aug = RandomApply(torchaudio.transforms.TimeMasking(time_mask_param=time_mask), p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)
