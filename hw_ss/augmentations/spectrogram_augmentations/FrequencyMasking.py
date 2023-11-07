import torchaudio
from torch import Tensor

from hw_ss.augmentations.base import AugmentationBase
from hw_ss.augmentations.random_apply import RandomApply


class FrequencyMasking(AugmentationBase):
    def __init__(self, frequency_mask, p):
        self._aug = RandomApply(torchaudio.transforms.FrequencyMasking(freq_mask_param=frequency_mask), p=p)

    def __call__(self, data: Tensor):
        return self._aug(data)
