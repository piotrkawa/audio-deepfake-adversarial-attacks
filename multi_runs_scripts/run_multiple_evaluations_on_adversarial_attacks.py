import argparse
import subprocess

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to batch config.", default="adv_batch_config.yaml")

    parser.add_argument(
        "--data_path",
        help="Dataset path",
        type=str,
        default="/home/adminuser/storage/datasets/deep_fakes",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as file:
        configs_to_run = yaml.safe_load(file)

    print("Running following configs: ", configs_to_run)

    for c in configs_to_run:
        attack = c["attack"]
        config = c["config"]
        attack_model_config = c["attack_model_config"]

        command = [
            f"python evaluate_models_on_adversarial_attacks.py "
            f"--attack {attack}",
            f"--config {config}",
            f"--attack_model_config {attack_model_config}",
            "--qual",
            "--raw_from_dataset"
            f"--asv_path {args.data_path}/ASVspoof2021/DF",
            f"--wavefake_path {args.data_path}/WaveFake",
            f"--celeb_path {args.data_path}/FakeAVCeleb/FakeAVCeleb_v1.2",
        ]

        command = " ".join(command)
        print(command)

        subprocess.call(command, shell=True)
