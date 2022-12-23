import argparse
import logging
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml
from torch import nn

from adversarial_attacks_generator.aa_trainer_types import AdversarialGDTrainerEnum
from src.datasets.detection_dataset import DetectionDataset
from src.models import models
from src.trainer import save_model
from src.utils import set_seed, load_model

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def get_datasets(
    datasets_paths: List[Union[Path, str]],
    amount_to_use: Optional[Tuple[int, int]] = None
) -> Tuple[DetectionDataset, DetectionDataset]:
    data_train = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        wavefake_path=datasets_paths[1],
        fakeavceleb_path=datasets_paths[2],
        subset="train",
        reduced_number=amount_to_use[0],
        oversample=True,
    )

    data_test = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        wavefake_path=datasets_paths[1],
        fakeavceleb_path=datasets_paths[2],
        subset="test",
        reduced_number=amount_to_use[1],
        oversample=True,
    )
    return data_train, data_test


def train_nn(
    datasets_paths: List[Union[Path, str]],
    batch_size: int,
    epochs: int,
    device: str,
    config: Dict,
    attack_config: Optional[Dict],
    adversarial_attacks: List[str],
    model_dir: Optional[Path] = None,
    amount_to_use: Optional[Tuple[int, int]] = None,
    config_save_path: str = "configs",
    adv_training_strategy: str = AdversarialGDTrainerEnum.RANDOM.name,
    is_finetune: bool = False,
) -> None:

    model_config = config["model"]
    model_name = model_config["name"]
    optimizer_config = model_config["optimizer"]

    LOGGER.info("Loading data...")

    timestamp = time.time()
    checkpoint_paths = []

    data_train, data_test = get_datasets(
        datasets_paths=datasets_paths,
        amount_to_use=amount_to_use,
    )

    current_model = models.get_model(
        model_name=model_name,
        config=model_config["parameters"],
        device=device,
    )

    if is_finetune:
        assert config["checkpoint"]["path"], "Finetune requires to provide checkpoint"

        weights_path = config["checkpoint"]["path"]
        lr = config["model"]["optimizer"]["lr"]
        LOGGER.info(f"Adversarial finetuning! Architecture: '{model_name}', lr: {lr}, weights: '{weights_path}'!")
        current_model.load_state_dict(torch.load(weights_path))

    current_model = current_model.to(device)
    current_model = nn.DataParallel(current_model)

    use_scheduler = "rawnet3" in model_name.lower()

    if attack_config is not None:
        LOGGER.info(f"Load attack model based on attack config")
        # If we dot provde attack_config - then we prepare attacks using the trained model.
        attack_model_name = attack_config["model"]["name"]
        attack_model = load_model(attack_config, device)
        attack_model = nn.DataParallel(attack_model)
        attack_info = f"{attack_model_name} (pretrained) {attack_config['checkpoint']['paths'][0]}"
    else:
        LOGGER.info(f"Use target model as attack model")
        attack_model = current_model
        attack_info = model_name

    LOGGER.info(f"Training '{model_name}', attacking using: '{attack_info}' model on {len(data_train)} audio files.")
    LOGGER.info(f"Adversarial training strategy: {adv_training_strategy}")

    save_name = f"aad__{model_name}_{timestamp}"

    current_model = AdversarialGDTrainerEnum[adv_training_strategy].value(
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_kwargs=optimizer_config,
        use_scheduler=use_scheduler,
    ).train(
        dataset=data_train,
        model=current_model,
        attack_model=attack_model,
        test_dataset=data_test,
        adversarial_attacks=adversarial_attacks,
        model_dir=model_dir,
        save_model_name=save_name
    )

    if model_dir is not None:
        save_model(
            model=current_model,
            model_dir=model_dir,
            name=save_name,
        )
        checkpoint_paths.append(str(model_dir.resolve() / save_name / "ckpt.pth"))

    LOGGER.info(f"Training model done!")

    # Save config for testing
    if model_dir is not None:
        config["checkpoint"] = {"paths": checkpoint_paths}
        config_name = f"aad__{model_name}__{timestamp}.yaml"
        config_save_path = str(Path(config_save_path) / config_name)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        LOGGER.info(f"Test config saved at location '{config_save_path}'!")


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.attack_model_config is not None:
        with open(args.attack_model_config, "r") as f:
            attack_model_config = yaml.safe_load(f)
    else:
        attack_model_config = None

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_nn(
        datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
        device=device,
        amount_to_use=(args.train_amount, args.test_amount),
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=model_dir,
        config=config,
        attack_config=attack_model_config,
        adversarial_attacks=config["data"].get("adversarial_attacks", []),
        adv_training_strategy=args.adv_training_strategy,
        is_finetune=args.finetune,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    ASVSPOOF_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/ASVspoof2021/DF"
    WAVEFAKE_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/WaveFake"
    FAKEAVCELEB_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/FakeAVCeleb/FakeAVCeleb_v1.2"

    parser.add_argument(
        "--asv_path",
        type=str,
        default=ASVSPOOF_DATASET_PATH,
        help="Path to ASVspoof2021 dataset directory",
    )
    parser.add_argument(
        "--wavefake_path",
        type=str,
        default=WAVEFAKE_DATASET_PATH,
        help="Path to WaveFake dataset directory",
    )
    parser.add_argument(
        "--celeb_path",
        type=str,
        default=FAKEAVCELEB_DATASET_PATH,
        help="Path to FakeAVCeleb dataset directory",
    )

    default_model_config = "config.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=default_model_config,
    )

    parser.add_argument(
        "--attack_model_config",
        help="Model config file path - if not provided, training will proceed using weights of the trained model",
        type=str,
        default=None,
    )

    default_train_amount = 100_000
    parser.add_argument(
        "--train_amount",
        "-a",
        help=f"Amount of files to load for training.",
        type=int,
        default=default_train_amount,
    )

    default_test_amount = 10_000
    parser.add_argument(
        "--test_amount",
        "-ta",
        help=f"Amount of files to load for testing.",
        type=int,
        default=default_test_amount,
    )

    default_batch_size = 64
    parser.add_argument(
        "--batch_size",
        "-b",
        help=f"Batch size (default: {default_batch_size}).",
        type=int,
        default=default_batch_size,
    )

    default_epochs = 5
    parser.add_argument(
        "--epochs",
        "-e",
        help=f"Epochs (default: {default_epochs}).",
        type=int,
        default=default_epochs,
    )

    default_model_dir = "trained_models"
    parser.add_argument(
        "--ckpt",
        help=f"Checkpoint directory (default: {default_model_dir}).",
        type=str,
        default=default_model_dir,
    )

    parser.add_argument(
        "--adv_training_strategy",
        help="Adversarial training strategy",
        type=str,
        default=AdversarialGDTrainerEnum.RANDOM.name,
        choices=[e.name for e in AdversarialGDTrainerEnum],
    )

    parser.add_argument("--cpu", "-c", help="Force using cpu?", action="store_true")

    parser.add_argument(
        "--verbose", "-v", help="Display debug information?", action="store_true"
    )

    parser.add_argument(
        "--finetune", help="Finetune using checkpoint provided in a config", action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
