from pathlib import Path
import logging

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset


DF_ASVSPOOF_KFOLD_SPLIT = {
    -1: {
        "partition_ratio": [0.7, 0.15],
        "seed": 45
    }
}

LOGGER = logging.getLogger()

class DeepFakeASVSpoofDatasetNoFold(SimpleAudioFakeDataset):

    protocol_file_name = "keys/CM/trial_metadata.txt"
    subset_dir_prefix = "ASVspoof2021_DF_eval"
    subset_parts = ("part00", "part01", "part02", "part03")

    def __init__(self, path, fold_num=-1, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = path

        if fold_num != -1:
            raise NotImplementedError

        self.partition_ratio = DF_ASVSPOOF_KFOLD_SPLIT[fold_num]["partition_ratio"]
        self.seed = DF_ASVSPOOF_KFOLD_SPLIT[fold_num]["seed"]

        self.flac_paths = self.get_file_references()
        self.samples = self.read_protocol()

        self.transform = transform
        # LOGGER.info(f"Spoof: {len(self.samples[self.samples['label'] == 'spoof'])}")
        # LOGGER.info(f"Original: {len(self.samples[self.samples['label'] == 'bonafide'])}")

    def get_file_references(self):
        flac_paths = {}
        for part in self.subset_parts:
            path = Path(self.path) / f"{self.subset_dir_prefix}_{part}" / self.subset_dir_prefix / "flac"
            flac_list = list(path.glob("*.flac"))

            for path in flac_list:
                flac_paths[path.stem] = path

        return flac_paths

    def read_protocol(self):
        samples = {
            "sample_name": [],
            "label": [],
            "path": []
        }

        real_samples = []
        fake_samples = []
        with open(Path(self.path) / self.protocol_file_name, "r") as file:
            for line in file:
                label = line.strip().split(" ")[5]

                if label == "bonafide":
                    real_samples.append(line)
                elif label == "spoof":
                    fake_samples.append(line)

        fake_samples = self.split_samples(fake_samples)
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line)

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line)

        return pd.DataFrame(samples)

    def add_line_to_samples(self, samples, line):
        _, sample_name, _, _, _, label, _, _ = line.strip().split(" ")
        samples["sample_name"].append(sample_name)
        samples["label"].append(label)

        sample_path = self.flac_paths[sample_name]
        assert sample_path.exists()
        samples["path"].append(sample_path)

        return samples


if __name__ == "__main__":
    ASVSPOOF_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/ASVspoof2021/DF"

    real = 0
    fake = 0
    datasets = []

    for subset in ['train', 'test', 'val']:
        dataset = DeepFakeASVSpoofDatasetNoFold(ASVSPOOF_DATASET_PATH, fold_num=-1, fold_subset=subset)

        real_samples = dataset.samples[dataset.samples['label'] == 'bonafide']
        real += len(real_samples)

        print('real', len(real_samples))

        spoofed_samples = dataset.samples[dataset.samples['label'] == 'spoof']
        fake += len(spoofed_samples)

        print('fake', len(spoofed_samples))

        datasets.append(dataset)

    paths_0 = [str(p) for p in datasets[0].samples.path.values]  # pathlib -> str
    paths_1 = [str(p) for p in datasets[1].samples.path.values]
    paths_2 = [str(p) for p in datasets[2].samples.path.values]

    assert len(paths_0) == len(set(paths_0)), "duplicated paths in subset"
    assert len(paths_1) == len(set(paths_1)), "duplicated paths in subset"
    assert len(paths_2) == len(set(paths_2)), "duplicated paths in subset"

    assert len(set(paths_0).intersection(set(paths_1))) == 0, "duplicated paths"
    assert len(set(paths_1).intersection(set(paths_2))) == 0, "duplicated paths"
    assert len(set(paths_0).intersection(set(paths_2))) == 0, "duplicated paths"

    print("All correct!")

    # TODO(PK): points to fulfill
    # [ ] each attack type should be present in each subset
    # [x] no duplicates
