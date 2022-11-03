from pathlib import Path

import pandas as pd

from dfadetect.agnostic_datasets.base_dataset import SimpleAudioFakeDataset


ASVSPOOF_KFOLD_SPLIT = {
    0: {
        "train": ['A01', 'A02', 'A03', 'A04', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A19'],
        "test": ['A05', 'A15', 'A16'],
        "val": ['A06', 'A17', 'A18'],
        "partition_ratio": [0.7, 0.15],
        "seed": 42
    },
    1: {
        "train": ['A01', 'A02', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'A15', 'A16', 'A17', 'A18', 'A19'],
        "test": ['A03', 'A11', 'A12'],
        "val": ['A04', 'A13', 'A14'],
        "partition_ratio": [0.7, 0.15],
        "seed": 43
    },
    2: {
        "train": ['A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
        "test": ['A01', 'A07', 'A08'],
        "val": ['A02', 'A09', 'A10'],
        "partition_ratio": [0.7, 0.15],
        "seed": 44
    },
    -1: {
        "train": ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
        "test":  ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
        "val":   ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
        "partition_ratio": [0.7, 0.15],
        "seed": 45
    }
}


class ASVSpoofDataset(SimpleAudioFakeDataset):

    protocol_folder_name = "ASVspoof2019_LA_cm_protocols"
    subset_dir_prefix = "ASVspoof2019_LA_"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = path

        self.allowed_attacks = ASVSPOOF_KFOLD_SPLIT[fold_num][fold_subset]
        self.partition_ratio = ASVSPOOF_KFOLD_SPLIT[fold_num]["partition_ratio"]
        self.seed = ASVSPOOF_KFOLD_SPLIT[fold_num]["seed"]

        self.samples = pd.DataFrame()

        for subset in self.subsets:
            subset_dir = Path(self.path) / f"{self.subset_dir_prefix}{subset}"
            subset_protocol_path = self.get_protocol_path(subset)
            subset_samples = self.read_protocol(subset_dir, subset_protocol_path)

            self.samples = pd.concat([self.samples, subset_samples])

        # self.samples, self.attack_signatures = self.group_by_attack()
        self.transform = transform

    def get_protocol_path(self, subset):
        paths = list((Path(self.path) / self.protocol_folder_name).glob("*.txt"))
        for path in paths:
            if subset in Path(path).stem:
                return path

    def read_protocol(self, subset_dir, protocol_path):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        real_samples = []
        fake_samples = []
        with open(protocol_path, "r") as file:
            for line in file:
                attack_type = line.strip().split(" ")[3]

                if attack_type == "-":
                    real_samples.append(line)
                elif attack_type in self.allowed_attacks:
                    fake_samples.append(line)

                if attack_type not in self.allowed_attacks:
                    continue

        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        return pd.DataFrame(samples)

    @staticmethod
    def add_line_to_samples(samples, line, subset_dir):
        user_id, sample_name, _, attack_type, label = line.strip().split(" ")
        samples["user_id"].append(user_id)
        samples["sample_name"].append(sample_name)
        samples["attack_type"].append(attack_type)
        samples["label"].append(label)

        assert (subset_dir / "flac" / f"{sample_name}.flac").exists()
        samples["path"].append(subset_dir / "flac" / f"{sample_name}.flac")

        return samples


class ASVSpoofDatasetNoFold(ASVSpoofDataset):

    def read_protocol(self, subset_dir, protocol_path):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        real_samples = []
        fake_samples = []
        with open(protocol_path, "r") as file:
            for line in file:
                attack_type = line.strip().split(" ")[3]

                if attack_type == "-":
                    real_samples.append(line)
                elif attack_type in self.allowed_attacks:
                    fake_samples.append(line)

                if attack_type not in self.allowed_attacks:
                    continue

        fake_samples = self.split_samples(fake_samples)
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        return pd.DataFrame(samples)


if __name__ == "__main__":
    ASVSPOOF_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/ASVspoof2021/LA"

    real = 0
    fake = 0
    datasets = []

    for subset in ['train', 'test', 'val']:
        dataset = ASVSpoofDatasetNoFold(ASVSPOOF_DATASET_PATH, fold_num=-1, fold_subset=subset)
        # print(dataset.samples["attack_type"].unique())
        # print(dataset.samples)

        real_samples = dataset.samples[dataset.samples['label'] == 'bonafide']
        real += len(real_samples)

        print('real', len(real_samples))

        spoofed_samples = dataset.samples[dataset.samples['label'] == 'spoof']
        fake += len(spoofed_samples)

        print('fake', len(spoofed_samples))
        datasets.append(dataset)

    print(real, fake)

    paths_0 = [str(p) for p in datasets[0].samples.path.values]  # pathlib -> str
    paths_1 = [str(p) for p in datasets[1].samples.path.values]
    paths_2 = [str(p) for p in datasets[2].samples.path.values]

    assert len(set(paths_0).intersection(set(paths_1))) == 0, "duplicated paths"
    assert len(set(paths_1).intersection(set(paths_2))) == 0, "duplicated paths"
    assert len(set(paths_0).intersection(set(paths_2))) == 0, "duplicated paths"

    print("All correct!")

    # TODO(PK): points to fulfill
    # [ ] each attack type should be present in each subset
    # [x] no duplicates
