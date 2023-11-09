from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from hw_ss.base.base_metric import BaseMetric
import torch


class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')

    def __call__(self, s1, s2, s3, audio_tgt, *args, **kwargs):
        with torch.no_grad():
            pred_audio = 0.8 * s1 + 0.1 * s2 + 0.1 * s3
            return self.pesq(pred_audio, audio_tgt.squeeze(1))
