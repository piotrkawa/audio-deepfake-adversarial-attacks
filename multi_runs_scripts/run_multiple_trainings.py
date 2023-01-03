import argparse
import subprocess

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to batch config.", default="batch_config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as file:
        batched_config = yaml.safe_load(file)
    configs_to_run = batched_config["paths"]

    print("Running following configs: ", configs_to_run)

    for current_config in configs_to_run:
        command = [
            f"python train_models.py "
            "--batch_size", "128",
            "--epochs", "10",
            f"--config {current_config}",
        ]

        command = " ".join(command)
        print(command)

        subprocess.call(command, shell=True)
