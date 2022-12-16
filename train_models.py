import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import yaml

from src.datasets.attack_agnostic_dataset import AttackAgnosticDataset, NoFoldDataset
from src.models import models
from src.trainer import GDTrainer
from src.utils import set_seed

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")


def get_datasets(
    datasets_paths: List[Union[Path, str]],
    fold: int,
    amount_to_use: Optional[int]
    ) -> Union[Tuple[NoFoldDataset, NoFoldDataset], Tuple[AttackAgnosticDataset, AttackAgnosticDataset]]:
    if fold == -1:
        data_train = NoFoldDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="train",
            reduced_number=100_000,
            oversample=True,
        )
        data_test = NoFoldDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="test",
            reduced_number=10_000,
            oversample=True,
        )
    else:
        data_train = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="train",
            reduced_number=amount_to_use,
            oversample=True,
        )

        data_test = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="test",
            reduced_number=amount_to_use,
            oversample=True,
        )

    return data_train, data_test


def train_nn(
    datasets_paths: List[Union[Path, str]],
    batch_size: int,
    epochs: int,
    device: str,
    config: Dict,
    model_dir: Optional[Path] = None,
    amount_to_use: Optional[int] = None,
    no_fold: bool = False,
    config_save_path: str = "configs",
) -> None:

    LOGGER.info("Loading data...")
    model_config = config["model"]
    model_name, model_parameters = model_config["name"], model_config["parameters"]
    optimizer_config = model_config["optimizer"]

    timestamp = time.time()
    checkpoint_paths = []
    folds = [0, 1, 2] if not no_fold else [-1]

    for fold in folds:

        data_train, data_test = get_datasets(
            datasets_paths=datasets_paths,
            fold=fold,
            amount_to_use=amount_to_use,
        )

        current_model = models.get_model(
            model_name=model_name,
            config=model_parameters,
            device=device,
        ).to(device)

        use_scheduler = "rawnet3" in model_name.lower()

        LOGGER.info(f"Training '{model_name}' model on {len(data_train)} audio files.")

        current_model = GDTrainer(
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_kwargs=optimizer_config,
            use_scheduler=use_scheduler,
        ).train(
            dataset=data_train,
            model=current_model,
            test_dataset=data_test,
            logging_prefix=f"fold_{fold}",
        )

        if model_dir is not None:
            save_name = f"aad__{model_name}_fold_{fold}__{timestamp}"
            save_model(
                model=current_model,
                model_dir=model_dir,
                name=save_name,
            )
            checkpoint_paths.append(str(model_dir.resolve() / save_name / "ckpt.pth"))

        LOGGER.info(f"Training model on fold [{fold+1}/{len(folds)}] done!")
    metrics_logger[f"parameters/data/save_paths"] = checkpoint_paths

    # Save config for testing
    if model_dir is not None:
        config["checkpoint"] = {"paths": checkpoint_paths}
        config_name = f"aad__{model_name}__{timestamp}.yaml"
        config_save_path = str(Path(config_save_path) / config_name)
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        metrics_logger["source_code/config"].upload(config_save_path)
        LOGGER.info(f"Test config saved at location '{config_save_path}'!")


def main(args):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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
        amount_to_use=args.amount,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_dir=model_dir,
        config=config,
        no_fold=args.no_fold,
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

    default_amount = None
    parser.add_argument(
        "--amount",
        "-a",
        help=f"Amount of files to load - useful when debugging (default: {default_amount} - use all).",
        type=int,
        default=default_amount,
    )

    default_batch_size = 128
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

    parser.add_argument("--cpu", "-c", help="Force using cpu?", action="store_true")

    parser.add_argument(
        "--verbose", "-v", help="Display debug information?", action="store_true"
    )

    parser.add_argument(
        "--no_fold", help="Use no fold version of the dataset", action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
