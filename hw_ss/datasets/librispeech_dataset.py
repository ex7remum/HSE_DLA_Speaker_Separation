import json
import logging
import os
import shutil
from pathlib import Path
import glob
from glob import glob

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm
import re

from hw_ss.base.base_dataset import BaseDataset
from hw_ss.utils import ROOT_PATH
from hw_ss.datasets.LibriSpeechSpeakerFiles import LibriSpeechSpeakerFiles
from hw_ss.datasets.MixtureGenerator import MixtureGenerator

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDataset(BaseDataset):
    def __init__(self, part, num_speakers, dataset_size, audio_len, data_dir=None, *args, **kwargs):
        assert part in URL_LINKS or part == 'train_all'
        self.num_speakers = num_speakers
        self.dataset_size = dataset_size
        self.audio_len = audio_len
        if data_dir is None:
            if "test" in part:
                add = "test"
            else:
                add = "train"
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech" / add
            data_dir_mix = data_dir / 'mix'
            data_dir_refs = data_dir / 'refs'
            data_dir_targets = data_dir / 'targets'
            data_dir.mkdir(exist_ok=True, parents=True)
            data_dir_mix.mkdir(exist_ok=True, parents=True)
            data_dir_refs.mkdir(exist_ok=True, parents=True)
            data_dir_targets.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, part):
        index = []

        if not os.listdir(os.path.join(self._data_dir, 'refs')):
            split_dir = Path("/kaggle/input/librispeech") / part / 'LibriSpeech' / part
            # split_dir = Path("/home/jupyter/mnt/datasets/LibriSpeech/LibriSpeech") / part
            # split_dir = self._data_dir / part
            if not split_dir.exists():
                self._load_part(part)

            speaker_ids = [speaker_id.name for speaker_id in os.scandir(split_dir)][:self.num_speakers]
            self.id2ind = {}
            self.ind2id = {}

            speaker_files = []
            for speaker_id in speaker_ids:
                speaker_files.append(LibriSpeechSpeakerFiles(speaker_id, split_dir,
                                                             audioTemplate='*.flac'))

            not_test = ('train' in part) or ('dev' in part)
            if not_test:
                mixture_generator = MixtureGenerator(speaker_files,
                                                     self._data_dir,
                                                     nfiles=self.dataset_size,
                                                     test=False)
                trim_db = None
                snr_levels = [-5, 5]
            else:
                mixture_generator = MixtureGenerator(speaker_files,
                                                     self._data_dir,
                                                     nfiles=self.dataset_size,
                                                     test=True)
                trim_db = None
                snr_levels = [0, 0]

            mixture_generator.generate_mixes(snr_levels=snr_levels,
                                             num_workers=2,
                                             update_steps=100,
                                             trim_db=trim_db,
                                             vad_db=20,
                                             audioLen=self.audio_len)

        # mixes = os.listdir(self._data_dir)
        all_ref = sorted(glob(os.path.join(self._data_dir, 'refs', '*-ref.wav')))
        all_mix = sorted(glob(os.path.join(self._data_dir, 'mix', '*-mixed.wav')))
        all_target = sorted(glob(os.path.join(self._data_dir, 'targets', '*-target.wav')))

        for ref, mix, target in zip(all_ref, all_mix, all_target):
            file_name = re.split('/', ref)[-1]
            parts_splitted = re.split('_', file_name)
            target_id, noise_id = parts_splitted[0], parts_splitted[1]
            index.append(
                {
                    "ref_path": ref,
                    "mix_path": mix,
                    "target_path": target,
                    "target_id": target_id
                }
            )
        return index
