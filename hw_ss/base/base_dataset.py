import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from hw_ss.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            limit=None,
            max_audio_length=None,
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs

        self._index = self._filter_records_from_dataset(index, limit, max_audio_length)

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path_tgt = data_dict["target_path"]
        audio_path_ref = data_dict["ref_path"]
        audio_path_mix = data_dict["mix_path"]

        audio_wave_tgt = self.load_audio(audio_path_tgt)
        audio_wave_ref = self.load_audio(audio_path_ref)
        audio_wave_mix = self.load_audio(audio_path_mix)

        return {
            'audio_tgt': audio_wave_tgt,
            'path_tgt': audio_path_tgt,
            'audio_ref': audio_wave_ref,
            'path_ref': audio_path_ref,
            'audio_mix': audio_wave_mix,
            'path_mix': audio_path_mix,
            'target_id': data_dict['target_id']
        }

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            wave2spec = self.config_parser.init_obj(
                self.config_parser["preprocessing"]["spectrogram"],
                torchaudio.transforms,
            )
            audio_tensor_spec = wave2spec(audio_tensor_wave)
            if self.spec_augs is not None:
                audio_tensor_spec = self.spec_augs(audio_tensor_spec)
            if self.log_spec:
                audio_tensor_spec = torch.log(audio_tensor_spec + 1e-5)
            return audio_tensor_wave, audio_tensor_spec

    @staticmethod
    def _filter_records_from_dataset(
            index: list, limit, max_audio_length
    ) -> list:
        initial_size = len(index)
        if max_audio_length is not None:
            audio_lengths = []
            for item in index:
                audio_tensor, sr = torchaudio.load(item['ref_path'])
                audio_tensor = audio_tensor[0:1, :]
                target_sr = 16000
                if sr != target_sr:
                    audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
                audio_lengths.append(audio_tensor.shape[-1])
            exceeds_audio_length = np.array(audio_lengths) >= max_audio_length
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        records_to_filter = exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(54)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index
