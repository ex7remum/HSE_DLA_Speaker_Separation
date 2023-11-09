from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from hw_ss.base.base_metric import BaseMetric
import torch


class SiSdrMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, s1, s2, s3, audio_tgt, *args, **kwargs):
        with torch.no_grad():
            pred_audio = 0.8 * s1 + 0.1 * s2 + 0.1 * s3
            return scale_invariant_signal_distortion_ratio(pred_audio, audio_tgt.squeeze(1), zero_mean=True).mean()
