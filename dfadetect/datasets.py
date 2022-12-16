"""Common preprocessing functions for audio data."""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torchaudio
from torchaudio.functional import apply_codec

from dfadetect.utils import find_wav_files

LOGGER = logging.getLogger(__name__)


SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]


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
        elif isinstance(directory_or_path_list, Path) \
                or isinstance(directory_or_path_list, str):
            directory = Path(directory_or_path_list)
            if not directory.exists():
                raise IOError(f"Directory does not exists: {self.directory}")

            paths = find_wav_files(directory)
            if paths is None:
                raise IOError(
                    f"Directory did not contain wav files: {self.directory}")
        else:
            raise TypeError(
                f"Supplied unsupported type for argument directory_or_path_list {type(directory_or_path_list)}!")

        if amount is not None:
            paths = paths[:amount]

        self._paths = paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = self._paths[index]

        waveform, sample_rate = torchaudio.load(path, normalize=self.normalize)

        if sample_rate != self.sample_rate:
            waveform, sample_rate = self.resample(path, self.sample_rate, self.normalize)

        if self.trim:
            waveform, sample_rate = self.apply_trim(waveform, sample_rate)

        if self.phone_call:
            waveform, sample_rate = self.process_phone_call(waveform, sample_rate)

        return waveform, sample_rate

    @staticmethod
    def apply_trim(waveform, sample_rate):
        waveform_trimmed, sample_rate_trimmed = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, SOX_SILENCE)

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
            path, [["rate", f"{target_sample_rate}"]], normalize=normalize)

        return waveform, sample_rate

    @staticmethod
    def process_phone_call(waveform, sample_rate):
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform,
            sample_rate,
            effects=[
                ["lowpass", "4000"],
                ["compand", "0.02,0.05",
                 "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8", "-7", "0.05"],
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

