import argparse
import subprocess

import yaml

from adversarial_attacks_generator.adversarial_training_types import \
    AdversarialGDTrainerEnum


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to batch config.", default="adv_batch_config.yaml"
    )
    parser.add_argument(
        "--adv_training_strategy",
        help="Adversarial training strategy",
        type=str,
        default=AdversarialGDTrainerEnum.RANDOM.name,
        choices=[e.name for e in AdversarialGDTrainerEnum],
    )

    parser.add_argument(
        "--data_path",
        help="Dataset path",
        type=str,
        default="/home/adminuser/storage/datasets/deep_fakes",
    )

    parser.add_argument(
        "--finetune", "-v", help="Use finetuning", action="store_true"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r") as file:
        configs_to_run = yaml.safe_load(file)

    print("Running following configs: ", configs_to_run)

    for c in configs_to_run:
        config = c["config"]
        attack_model_config = c.get("attack_model_config", None)

        command = [
            f"python train_models_with_adversarial_attacks.py "
            f"--config {config}",
            "--epochs 10",
            f"--adv_training_strategy {args.adv_training_strategy}",
            f"--asv_path {args.data_path}/ASVspoof2021/DF",
            f"--wavefake_path {args.data_path}/WaveFake",
            f"--celeb_path {args.data_path}/FakeAVCeleb/FakeAVCeleb_v1.2",
        ]

        if attack_model_config is not None:
            command.append(
                f"--attack_model_config {attack_model_config}"
            )

        if args.finetune:
            command.append("--finetune")

        command = " ".join(command)
        print(command)

        subprocess.call(command, shell=True)
