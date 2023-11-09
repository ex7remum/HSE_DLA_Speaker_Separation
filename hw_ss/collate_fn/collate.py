import logging
from typing import List
import torch
import numpy as np

logger = logging.getLogger(__name__)


def form_batch(dataset_items: List[dict], audio_type: str):
    batch_size = len(dataset_items)
    paths_name = 'path_' + audio_type
    audio_name = 'audio_' + audio_type
    audio_lengths = torch.tensor([item[audio_name].shape[1] for item in dataset_items])

    if audio_type == 'ref':
        for ref_idx, ref_length in enumerate(audio_lengths):
            if ref_length > 16000 * 7:
                audio_lengths[ref_idx] = 16000 * 7

    max_audio_time = audio_lengths.max()
    audios = torch.zeros((batch_size, 1, max_audio_time))
    for item_num, item in enumerate(dataset_items):
        cur_audio = item[audio_name]
        audios[item_num, :, :min(cur_audio.shape[1], max_audio_time)] = cur_audio

    paths = [item[paths_name] for item in dataset_items]
    return paths, audios, audio_lengths


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    path_ref, audio_ref, len_ref = form_batch(dataset_items, 'ref')
    path_mix, audio_mix, len_mix = form_batch(dataset_items, 'mix')
    path_tgt, audio_tgt, len_tgt = form_batch(dataset_items, 'tgt')

    target_ids = torch.tensor([item['target_id'] for item in dataset_items])

    result_batch = {'audio_tgt': audio_tgt,
                    'path_tgt': path_tgt,
                    'len_tgt': len_tgt,
                    'audio_ref': audio_ref,
                    'path_ref': path_ref,
                    'len_ref': len_ref,
                    'audio_mix': audio_mix,
                    'path_mix': path_mix,
                    'len_mix': len_mix,
                    'target_id': target_ids}
    return result_batch
