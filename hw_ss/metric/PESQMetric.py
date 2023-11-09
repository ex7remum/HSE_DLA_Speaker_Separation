from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from hw_ss.base.base_metric import BaseMetric
import torch


class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, s1, s2, s3, audio_tgt, *args, **kwargs):
        with torch.no_grad():
            pred_audio = 0.8 * s1 + 0.1 * s2 + 0.1 * s3
            return perceptual_evaluation_speech_quality(pred_audio, audio_tgt.squeeze(1), 16000, 'wb').mean()
