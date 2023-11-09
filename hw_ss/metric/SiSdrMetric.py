from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from hw_ss.base.base_metric import BaseMetric
import torch


class SiSdrMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

    def __call__(self, s1, s2, s3, audio_tgt, *args, **kwargs):
        with torch.no_grad():
            pred_audio = 0.8 * s1 + 0.1 * s2 + 0.1 * s3
            print(pred_audio.device, audio_tgt.device)
            return self.si_sdr(pred_audio, audio_tgt.squeeze(1))
