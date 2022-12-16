"""Base dataset classes."""
import logging
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchaudio.functional import apply_codec

from src.utils import find_wav_files

LOGGER = logging.getLogger(__name__)

WAVE_FAKE_INTERFACE = True
WAVE_FAKE_SR = 16_000
WAVE_FAKE_TRIM = True
WAVE_FAKE_NORMALIZE = True
WAVE_FAKE_CELL_PHONE = False
WAVE_FAKE_PAD = True
WAVE_FAKE_CUT = 64_600

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]


class SimpleAudioFakeDataset(Dataset):
    def __init__(
        self,
        fold_num,
        fold_subset,
        transform=None,
        return_label=True,
        return_meta=False,
        return_raw=False,
    ):
        self.transform = transform
        self.samples = pd.DataFrame()

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = None
        self.partition_ratio = None
        self.seed = None
        self.return_label = return_label
        self.return_meta = return_meta
        self.return_raw = return_raw

    def split_samples(self, samples_list):
        if isinstance(samples_list, pd.DataFrame):
            samples_list = samples_list.sort_values(by=list(samples_list.columns))
            samples_list = samples_list.sample(frac=1, random_state=self.seed)
        else:
            samples_list = sorted(samples_list)
            random.seed(self.seed)
            random.shuffle(samples_list)

        p, s = self.partition_ratio
        subsets = np.split(
            samples_list, [int(p * len(samples_list)), int((p + s) * len(samples_list))]
        )
        return dict(zip(["train", "test", "val"], subsets))[self.fold_subset]

    def df2tuples(self):
        tuple_samples = []
        for i, elem in self.samples.iterrows():
            tuple_samples.append(
                (str(elem["path"]), elem["label"], elem["attack_type"])
            )

        self.samples = tuple_samples
        return self.samples

    @staticmethod
    def wavefake_preprocessing(
        waveform,
        sample_rate,
        wave_fake_sr: Optional[int] = None,
        wave_fake_trim: Optional[bool] = None,
        wave_fake_cell_phone: Optional[bool] = None,
        wave_fake_pad: Optional[bool] = None,
        wave_fake_cut: Optional[int] = None,
    ):
        wave_fake_sr = WAVE_FAKE_SR if wave_fake_sr is None else wave_fake_sr
        wave_fake_trim = WAVE_FAKE_TRIM if wave_fake_trim is None else wave_fake_trim
        wave_fake_cell_phone = (
            WAVE_FAKE_CELL_PHONE
            if wave_fake_cell_phone is None
            else wave_fake_cell_phone
        )
        wave_fake_pad = WAVE_FAKE_PAD if wave_fake_pad is None else wave_fake_pad
        wave_fake_cut = WAVE_FAKE_CUT if wave_fake_cut is None else wave_fake_cut

        if sample_rate != wave_fake_sr and wave_fake_sr != -1:
            waveform, sample_rate = AudioDataset.resample_wave(
                waveform, sample_rate, wave_fake_sr
            )

        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform[:1, ...]

        if wave_fake_trim:
            waveform, sample_rate = AudioDataset.apply_trim(waveform, sample_rate)

        if wave_fake_cell_phone:
            waveform, sample_rate = AudioDataset.process_phone_call(
                waveform, sample_rate
            )

        if wave_fake_pad:
            waveform = PadDataset.apply_pad(waveform, wave_fake_cut)

        return waveform, sample_rate

    @staticmethod
    def wavefake_preprocessing_on_batch(
        batch_waveform, batch_sample_rate, *args, **kwargs
    ):
        device_waveform = torch.get_device(batch_waveform)
        device_waveform = f"cuda:{device_waveform}" if device_waveform >= 0 else "cpu"
        device_sample_rate = torch.get_device(batch_sample_rate)
        device_sample_rate = (
            f"cuda:{device_sample_rate}" if device_sample_rate >= 0 else "cpu"
        )

        batch_waveform = batch_waveform.cpu()
        batch_sample_rate = batch_sample_rate.cpu()

        waveforms = []
        sample_rates = []
        for waveform, sample_rate in zip(batch_waveform, batch_sample_rate):
            waveform = waveform.unsqueeze(0)
            waveform, sample_rate = SimpleAudioFakeDataset.wavefake_preprocessing(
                waveform, sample_rate, *args, **kwargs
            )
            waveforms.append(waveform)
            sample_rates.append(torch.tensor([sample_rate]))

        batch_waveform = torch.stack(waveforms, dim=0).to(device_waveform)
        batch_sample_rate = torch.cat(sample_rates, dim=0).to(device_sample_rate)
        return batch_waveform, batch_sample_rate

    def __getitem__(self, index) -> T_co:

        if isinstance(self.samples, pd.DataFrame):
            sample = self.samples.iloc[index]

            path = str(sample["path"])
            label = sample["label"]
            attack_type = sample["attack_type"]
            if type(attack_type) != str and math.isnan(attack_type):
                attack_type = "N/A"
        else:
            path, label, attack_type = self.samples[index]

        if WAVE_FAKE_INTERFACE:
            # TODO: apply normalization from torchaudio.load
            waveform, sample_rate = torchaudio.load(path, normalize=WAVE_FAKE_NORMALIZE)
            real_sec_length = len(waveform[0]) / sample_rate

            if self.return_raw:
                waveform, sample_rate = self.wavefake_preprocessing(
                    waveform,
                    sample_rate,
                    wave_fake_trim=False,
                    wave_fake_cell_phone=False,
                )
            else:
                waveform, sample_rate = self.wavefake_preprocessing(
                    waveform, sample_rate
                )

            return_data = [waveform, sample_rate]
            if self.return_label:
                label = 1 if label == "bonafide" else 0
                return_data.append(label)

            if self.return_meta:
                return_data.append(
                    (
                        attack_type,
                        path,
                        self.fold_num,
                        self.fold_subset,
                        real_sec_length,
                    )
                )
            return return_data

        # TODO: remove it, we do not use it
        data, sr = sf.read(path)

        if self.transform:
            data = self.transform(data)

        return data, label, attack_type

    def __len__(self):
        return len(self.samples)


class AudioDataset(torch.utils.data.Dataset):
    """Torch dataset to load data from a provided directory.

    Args:
        directory_or_path_list: Path to the directory containing wav files to load. Or a list of paths.
    Raises:
        IOError: If the directory does ot exists or the directory did not contain any wav files.
    """

    def __init__(
        self,
        directory_or_path_list: Union[Union[str, Path], List[Union[str, Path]]],
        sample_rate: int = 16_000,
        amount: Optional[int] = None,
        normalize: bool = True,
        trim: bool = True,
        phone_call: bool = False,
    ) -> None:
        super().__init__()

        self.trim = trim
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.phone_call = phone_call

        if isinstance(directory_or_path_list, list):
            paths = directory_or_path_list
        elif isinstance(directory_or_path_list, Path) or isinstance(
            directory_or_path_list, str
        ):
            directory = Path(directory_or_path_list)
            if not directory.exists():
                raise IOError(f"Directory does not exists: {self.directory}")

            paths = find_wav_files(directory)
            if paths is None:
                raise IOError(f"Directory did not contain wav files: {self.directory}")
        else:
            raise TypeError(
                f"Supplied unsupported type for argument directory_or_path_list {type(directory_or_path_list)}!"
            )

        if amount is not None:
            paths = paths[:amount]

        self._paths = paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self._paths[index]

        waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        if sample_rate != self.sample_rate:
            waveform, sample_rate = self.resample(
                path, self.sample_rate, self.normalize
            )

        if self.trim:
            waveform, sample_rate = self.apply_trim(waveform, sample_rate)

        if self.phone_call:
            waveform, sample_rate = self.process_phone_call(waveform, sample_rate)

        return waveform, sample_rate

    @staticmethod
    def apply_trim(waveform, sample_rate):
        (
            waveform_trimmed,
            sample_rate_trimmed,
        ) = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, SOX_SILENCE
        )

        if waveform_trimmed.size()[1] > 0:
            waveform = waveform_trimmed
            sample_rate = sample_rate_trimmed

        return waveform, sample_rate

    @staticmethod
    def resample_wave(waveform, sample_rate, target_sample_rate):
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
        )
        return waveform, sample_rate

    @staticmethod
    def resample(path, target_sample_rate, normalize=True):
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
            path, [["rate", f"{target_sample_rate}"]], normalize=normalize
        )

        return waveform, sample_rate

    @staticmethod
    def process_phone_call(waveform, sample_rate):
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            sample_rate,
            effects=[
                ["lowpass", "4000"],
                [
                    "compand",
                    "0.02,0.05",
                    "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8",
                    "-8",
                    "-7",
                    "0.05",
                ],
                ["rate", "8000"],
            ],
        )
        waveform = apply_codec(waveform, sample_rate, format="gsm")
        return waveform, sample_rate

    def __len__(self) -> int:
        return len(self._paths)


class PadDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, cut: int = 64600, label=None):
        self.dataset = dataset
        self.cut = cut  # max 4 sec (ASVSpoof default)
        self.label = label

    def __getitem__(self, index):
        waveform, sample_rate = self.dataset[index]
        waveform = self.apply_pad(waveform, self.cut)

        if self.label is None:
            return waveform, sample_rate
        else:
            return waveform, sample_rate, self.label

    @staticmethod
    def apply_pad(waveform, cut):
        waveform = waveform.squeeze(0)
        waveform_len = waveform.shape[0]

        if waveform_len >= cut:
            return waveform[:cut]

        # need to pad
        num_repeats = int(cut / waveform_len) + 1
        padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

        return padded_waveform

    def __len__(self):
        return len(self.dataset)
