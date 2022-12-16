from pathlib import Path

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset


FAKEAVCELEB_KFOLD_SPLIT = {
    0: {
        "train": ['rtvc', 'faceswap-wav2lip'],
        "test": ['fsgan-wav2lip'],
        "val": ['wav2lip'],
        "partition_ratio": [0.7, 0.15],
        "seed": 42
    },
    1: {
        "train": ['fsgan-wav2lip', 'wav2lip'],
        "test": ['rtvc'],
        "val": ['faceswap-wav2lip'],
        "partition_ratio": [0.7, 0.15],
        "seed": 43
    },
    2: {
        "train": ['faceswap-wav2lip', 'fsgan-wav2lip'],
        "test": ['wav2lip'],
        "val": ['rtvc'],
        "partition_ratio": [0.7, 0.15],
        "seed": 44
    },
    -1: {
        "train": ['faceswap-wav2lip', 'fsgan-wav2lip', 'wav2lip', 'rtvc'],
        "test":  ['faceswap-wav2lip', 'fsgan-wav2lip', 'wav2lip', 'rtvc'],
        "val":   ['faceswap-wav2lip', 'fsgan-wav2lip', 'wav2lip', 'rtvc'],
        "partition_ratio": [0.7, 0.15],
        "seed": 45
    },
}


class FakeAVCelebDataset(SimpleAudioFakeDataset):

    audio_folder = "FakeAVCeleb-audio"
    audio_extension = ".mp3"
    metadata_file = Path(audio_folder) / "meta_data.csv"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, fold_num=0, fold_subset="train", transform=None):
        super().__init__(fold_num, fold_subset, transform)
        self.path = path

        self.fold_num, self.fold_subset = fold_num, fold_subset
        self.allowed_attacks = FAKEAVCELEB_KFOLD_SPLIT[fold_num][fold_subset]
        self.partition_ratio = FAKEAVCELEB_KFOLD_SPLIT[fold_num]["partition_ratio"]
        self.seed = FAKEAVCELEB_KFOLD_SPLIT[fold_num]["seed"]

        self.metadata = self.get_metadata()

        self.samples = pd.concat([self.get_fake_samples(), self.get_real_samples()], ignore_index=True)

    def get_metadata(self):
        md = pd.read_csv(Path(self.path) / self.metadata_file)
        md["audio_type"] = md["type"].apply(lambda x: x.split("-")[-1])
        return md

    def get_fake_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        for attack_name in self.allowed_attacks:
            fake_samples = self.metadata[
                (self.metadata["method"] == attack_name) & (self.metadata["audio_type"] == "FakeAudio")
            ]

            samples_list = fake_samples.iterrows()

            for index, sample in samples_list:
                samples["user_id"].append(sample["source"])
                samples["sample_name"].append(Path(sample["filename"]).stem)
                samples["attack_type"].append(sample["method"])
                samples["label"].append("spoof")
                samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)

    def get_real_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        samples_list = self.metadata[
            (self.metadata["method"] == "real") & (self.metadata["audio_type"] == "RealAudio")
        ]

        samples_list = self.split_samples(samples_list)

        for index, sample in samples_list.iterrows():
            samples["user_id"].append(sample["source"])
            samples["sample_name"].append(Path(sample["filename"]).stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")
            samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)

    def get_file_path(self, sample):
        path = "/".join([self.audio_folder, *sample["path"].split("/")[1:]])
        return Path(self.path) / path / Path(sample["filename"]).with_suffix(self.audio_extension)


class FakeAVCelebDatasetNoFold(FakeAVCelebDataset):
    def get_fake_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        for attack_name in self.allowed_attacks:
            fake_samples = self.metadata[
                (self.metadata["method"] == attack_name) & (self.metadata["audio_type"] == "FakeAudio")
            ]

            samples_list = fake_samples.iterrows()
            samples_list = self.split_samples(samples_list)

            for index, sample in samples_list:
                samples["user_id"].append(sample["source"])
                samples["sample_name"].append(Path(sample["filename"]).stem)
                samples["attack_type"].append(sample["method"])
                samples["label"].append("spoof")
                samples["path"].append(self.get_file_path(sample))

        return pd.DataFrame(samples)


if __name__ == "__main__":
    FAKEAVCELEB_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/FakeAVCeleb/FakeAVCeleb_v1.2"

    real = 0
    fake = 0
    datasets = []

    for subset in ['train', 'test', 'val']:
        dataset = FakeAVCelebDatasetNoFold(FAKEAVCELEB_DATASET_PATH, fold_num=-1, fold_subset=subset)

        real_samples = dataset.samples[dataset.samples['label'] == 'bonafide']
        real += len(real_samples)
        print('real', len(real_samples))

        spoofed_samples = dataset.samples[dataset.samples['label'] == 'spoof']
        fake += len(spoofed_samples)
        print('fake', len(spoofed_samples))
        datasets.append(dataset)



    # Unique Calculation
    if False:
        print(real, fake)
        paths_0 = [str(p) for p in datasets[0].samples.path.values]  # pathlib -> str
        paths_1 = [str(p) for p in datasets[1].samples.path.values]
        paths_2 = [str(p) for p in datasets[2].samples.path.values]
        
        inter_1_2 = set(paths_1).intersection(set(paths_2))
        inter_0_1 = set(paths_0).intersection(set(paths_1))
        inter_0_2 = set(paths_0).intersection(set(paths_2))
        assert len(inter_1_2) == 0, "duplicated paths"
        assert len(inter_0_1) == 0, "duplicated paths"
        assert len(inter_0_2) == 0, "duplicated paths"
        # TODO(PK): there are few duplicates - investigate!

    print("All correct!")

    # import pandas as pd
    # df = pd.read_csv("../../meta_data.csv")
    # filepaths = df["filename"]
    # for type in df['type'].unique():
    #     df_type = df[df.type == type]
    #     samples = df_type.filename
    #     all_samples_count = len(samples)
    #     unique_samples = set(samples)
    #     unique_samples_count = len(unique_samples)
    #     print(all_samples_count, unique_samples_count)

    # filepaths = df["filepath"]
