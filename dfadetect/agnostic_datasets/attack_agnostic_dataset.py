import logging
import pandas as pd
from typing import List, Optional

from dfadetect.agnostic_datasets.asvspoof_dataset import ASVSpoofDataset
from dfadetect.agnostic_datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDatasetNoFold
from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset
from dfadetect.agnostic_datasets.fakeavceleb_dataset import FakeAVCelebDataset, FakeAVCelebDatasetNoFold
from dfadetect.agnostic_datasets.wavefake_dataset import WaveFakeDataset, WaveFakeDatasetNoFold

LOGGER = logging.getLogger()


class AttackAgnosticDataset(SimpleAudioFakeDataset):

    def __init__(
        self,
        asvspoof_path=None,
        wavefake_path=None,
        fakeavceleb_path=None,
        fold_num=0,
        fold_subset="val",
        transform=None,
        oversample=True,
        undersample=False,
        return_label=True,
        reduced_number=None,
        return_meta=False,
        return_raw=False
    ):
        super().__init__(fold_num, fold_subset, transform, return_label, return_meta, return_raw)
        datasets = self._init_datasets(
            asvspoof_path=asvspoof_path,
            wavefake_path=wavefake_path,
            fakeavceleb_path=fakeavceleb_path,
            fold_num=fold_num,
            fold_subset=fold_subset,
        )
        self.samples = pd.concat(
            [ds.samples for ds in datasets],
            ignore_index=True
        )

        if oversample:
            self.oversample_dataset()
        elif undersample:
            self.undersample_dataset()

        if reduced_number:
            LOGGER.info(f"Using reduced number of samples - {reduced_number}!")
            self.samples = self.samples.sample(
                min(len(self.samples), reduced_number),
                random_state=42,
            )

    def _init_datasets(
        self,
        asvspoof_path: Optional[str],
        wavefake_path: Optional[str],
        fakeavceleb_path: Optional[str],
        fold_num: int,
        fold_subset: str,
    ) -> List[SimpleAudioFakeDataset]:
        datasets = []

        if asvspoof_path is not None:
            asvspoof_dataset = ASVSpoofDataset(asvspoof_path, fold_num=fold_num, fold_subset=fold_subset)
            datasets.append(asvspoof_dataset)

        if wavefake_path is not None:
            wavefake_dataset = WaveFakeDataset(wavefake_path, fold_num=fold_num, fold_subset=fold_subset)
            datasets.append(wavefake_dataset)

        if fakeavceleb_path is not None:
            fakeavceleb_dataset = FakeAVCelebDataset(fakeavceleb_path, fold_num=fold_num, fold_subset=fold_subset)
            datasets.append(fakeavceleb_dataset)
        return datasets

    def oversample_dataset(self):
        samples = self.samples.groupby(by=['label'])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        diff_length = spoof_length - bona_length

        if diff_length < 0:
            raise NotImplementedError

        if diff_length > 0:
            bonafide = samples.get_group("bonafide").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)

    def undersample_dataset(self):
        samples = self.samples.groupby(by=['label'])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        if spoof_length < bona_length:
            raise NotImplementedError

        if spoof_length > bona_length:
            spoofs = samples.get_group("spoof").sample(bona_length, replace=True)
            self.samples = pd.concat([samples.get_group("bonafide"), spoofs], ignore_index=True)

    def get_bonafide_only(self):
        samples = self.samples.groupby(by=['label'])
        self.samples = samples.get_group("bonafide")
        return self.samples

    def get_spoof_only(self):
        samples = self.samples.groupby(by=['label'])
        self.samples = samples.get_group("spoof")
        return self.samples


class NoFoldDataset(AttackAgnosticDataset):
    def _init_datasets(
        self,
        asvspoof_path: Optional[str],
        wavefake_path: Optional[str],
        fakeavceleb_path: Optional[str],
        fold_num: int,
        fold_subset: str,
    ) -> List[SimpleAudioFakeDataset]:
        datasets = []

        if asvspoof_path is not None:
            asvspoof_dataset = DeepFakeASVSpoofDatasetNoFold(asvspoof_path, fold_num=fold_num, fold_subset=fold_subset)
            datasets.append(asvspoof_dataset)

        if wavefake_path is not None:
            wavefake_dataset = WaveFakeDatasetNoFold(wavefake_path, fold_num=fold_num, fold_subset=fold_subset)
            datasets.append(wavefake_dataset)

        if fakeavceleb_path is not None:
            fakeavceleb_dataset = FakeAVCelebDatasetNoFold(fakeavceleb_path, fold_num=fold_num, fold_subset=fold_subset)
            datasets.append(fakeavceleb_dataset)

        return datasets
