import math
import random
from typing import Optional

import numpy as np
import torch
import pandas as pd
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from dfadetect.datasets import AudioDataset, PadDataset

WAVE_FAKE_INTERFACE = True
WAVE_FAKE_SR = 16_000
WAVE_FAKE_TRIM = True
WAVE_FAKE_NORMALIZE = True
WAVE_FAKE_CELL_PHONE = False
WAVE_FAKE_PAD = True
WAVE_FAKE_CUT = 64_600


class SimpleAudioFakeDataset(Dataset):

    def __init__(self, fold_num, fold_subset, transform=None, return_label=True, return_meta=False, return_raw=False):
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
        subsets = np.split(samples_list, [int(p*len(samples_list)), int((p+s)*len(samples_list))])
        return dict(zip(['train', 'test', 'val'], subsets))[self.fold_subset]

    def df2tuples(self):
        tuple_samples = []
        for i, elem in self.samples.iterrows():
            tuple_samples.append((str(elem["path"]), elem["label"], elem["attack_type"]))

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
            wave_fake_cut: Optional[int] = None
    ):
        wave_fake_sr = WAVE_FAKE_SR if wave_fake_sr is None else wave_fake_sr
        wave_fake_trim = WAVE_FAKE_TRIM if wave_fake_trim is None else wave_fake_trim
        wave_fake_cell_phone = WAVE_FAKE_CELL_PHONE if wave_fake_cell_phone is None else wave_fake_cell_phone
        wave_fake_pad = WAVE_FAKE_PAD if wave_fake_pad is None else wave_fake_pad
        wave_fake_cut = WAVE_FAKE_CUT if wave_fake_cut is None else wave_fake_cut

        if sample_rate != wave_fake_sr and wave_fake_sr != -1:
            waveform, sample_rate = AudioDataset.resample_wave(waveform, sample_rate, wave_fake_sr)

        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform[:1, ...]

        if wave_fake_trim:
            waveform, sample_rate = AudioDataset.apply_trim(waveform, sample_rate)

        if wave_fake_cell_phone:
            waveform, sample_rate = AudioDataset.process_phone_call(waveform, sample_rate)

        if wave_fake_pad:
            waveform = PadDataset.apply_pad(waveform, wave_fake_cut)

        return waveform, sample_rate

    @staticmethod
    def wavefake_preprocessing_on_batch(batch_waveform, batch_sample_rate, *args, **kwargs):
        device_waveform = torch.get_device(batch_waveform)
        device_waveform = f"cuda:{device_waveform}" if device_waveform >= 0 else "cpu"
        device_sample_rate = torch.get_device(batch_sample_rate)
        device_sample_rate = f"cuda:{device_sample_rate}" if device_sample_rate >= 0 else "cpu"

        batch_waveform = batch_waveform.cpu()
        batch_sample_rate = batch_sample_rate.cpu()

        waveforms = []
        sample_rates = []
        for waveform, sample_rate in zip(batch_waveform, batch_sample_rate):
            waveform = waveform.unsqueeze(0)
            waveform, sample_rate = SimpleAudioFakeDataset.wavefake_preprocessing(waveform, sample_rate, *args, **kwargs)
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
                    wave_fake_cell_phone=False
                )
            else:
                waveform, sample_rate = self.wavefake_preprocessing(waveform, sample_rate)

            # leave it for now, to be sure that wavefake_preprocessing replicate it correctly
            # if sample_rate != WAVE_FAKE_SR:
            #     waveform, sample_rate = AudioDataset.resample(path, WAVE_FAKE_SR, WAVE_FAKE_NORMALIZE)
            #
            # if waveform.dim() > 1 and waveform.shape[0] > 1:
            #     waveform = waveform[:1, ...]
            #
            # if WAVE_FAKE_TRIM:
            #     waveform, sample_rate = AudioDataset.apply_trim(waveform, sample_rate)
            #
            # if WAVE_FAKE_CELL_PHONE:
            #     waveform, sample_rate = AudioDataset.process_phone_call(waveform, sample_rate)
            #
            # if WAVE_FAKE_PAD:
            #     waveform = PadDataset.apply_pad(waveform, WAVE_FAKE_CUT)

            return_data = [waveform, sample_rate]
            if self.return_label:
                label = 1 if label == "bonafide" else 0
                return_data.append(label)

            if self.return_meta:
                return_data.append((attack_type, path, self.fold_num, self.fold_subset, real_sec_length))
            return return_data

        # TODO: remove it, we do not use it
        data, sr = sf.read(path)

        if self.transform:
            data = self.transform(data)

        return data, label, attack_type

    def __len__(self):
        return len(self.samples)

