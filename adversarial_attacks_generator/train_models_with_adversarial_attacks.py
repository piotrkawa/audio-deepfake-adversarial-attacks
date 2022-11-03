import argparse
import logging
import sys
import time
from pathlib import Path
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


import torch
from torch import nn
import tqdm
import yaml

from dfadetect.agnostic_datasets.attack_agnostic_dataset import AttackAgnosticDataset, NoFoldDataset
from adversarial_attacks_generator.adversarial_training_types import AdversarialGDTrainerEnum
from dfadetect.cnn_features import CNNFeaturesSetting
from dfadetect.models import models
from dfadetect import neptune_utils
from dfadetect.trainer import NNDataSetting, save_model
from dfadetect.utils import set_seed

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


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
    attack_config: Optional[Dict],
    cnn_features_setting: CNNFeaturesSetting,
    metrics_logger,
    adversarial_attacks: List[str],
    model_dir: Optional[Path] = None,
    amount_to_use: Optional[int] = None,
    config_save_path: str = "configs",
    no_fold: bool = False,
    adv_training_strategy: str = AdversarialGDTrainerEnum.RANDOM.name,
    is_finetune: bool = False,
) -> None:

    model_config = config["model"]
    model_name = model_config["name"]
    optimizer_config = model_config["optimizer"]
    use_cnn_features = False if model_name in ["rawnet", "rawnet3", "frontend_lcnn", "frontend_specrnet"] else True

    LOGGER.info("Loading data...")
    nn_data_setting = NNDataSetting(
        use_cnn_features=use_cnn_features,
    )

    timestamp = time.time()
    checkpoint_paths = []
    folds = [0, 1, 2] if not no_fold else [-1]

    for fold_no, fold in enumerate(tqdm.tqdm(folds)):
        data_train, data_test = get_datasets(
            datasets_paths=datasets_paths,
            fold=fold,
            amount_to_use=amount_to_use,
        )

        current_model = models.get_model(
            model_name=model_name,
            config=model_config["parameters"],
            device=device,
        )

        if is_finetune:
            assert config["checkpoint"]["paths"][0], "Finetune requires to provide checkpoint"
            assert len(config["checkpoint"]["paths"]) == 1, "Only NO_FOLD is currently supported"

            weights_path = config["checkpoint"]["paths"][0]
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
            attack_model = load_model(attack_config, fold, device)
            attack_model = nn.DataParallel(attack_model)
            attack_info = f"{attack_model_name} (pretrained) {attack_config['checkpoint']['paths'][0]}"
        else:
            LOGGER.info(f"Use target model as attack model")
            attack_model = current_model
            attack_info = model_name

        LOGGER.info(f"Training '{model_name}', attacking using: '{attack_info}' model on {len(data_train)} audio files.")
        LOGGER.info(f"Adversarial training strategy: {adv_training_strategy}")

        save_name = f"aad__{model_name}_fold_{fold}__{timestamp}"

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
            nn_data_setting=nn_data_setting,
            logger=metrics_logger,
            logging_prefix=f"fold_{fold}",
            cnn_features_setting=cnn_features_setting,
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

    if args.attack_model_config is not None:
        with open(args.attack_model_config, "r") as f:
            attack_model_config = yaml.safe_load(f)
    else:
        attack_model_config = None

    seed = config["data"].get("seed", 42)
    # fix all seeds
    set_seed(seed)

    neptune_instance = neptune_init(args=args, config=config, attack_model_config=attack_model_config)

    if not args.cpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model_dir = Path(args.ckpt)
    model_dir.mkdir(parents=True, exist_ok=True)

    cnn_features_setting = config["data"].get("cnn_features_setting", None)
    if cnn_features_setting:
        cnn_features_setting = CNNFeaturesSetting(**cnn_features_setting)
    else:
        cnn_features_setting = CNNFeaturesSetting()

    train_nn(
        datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
        device=device,
        amount_to_use=args.amount,
        batch_size=args.batch_size,
        metrics_logger=neptune_instance,
        epochs=args.epochs,
        model_dir=model_dir,
        config=config,
        attack_config=attack_model_config,
        cnn_features_setting=cnn_features_setting,
        adversarial_attacks=config["data"].get("adversarial_attacks", []),
        no_fold=args.no_fold,
        adv_training_strategy=args.adv_training_strategy,
        is_finetune=args.finetune,
    )


def load_model(model_config, fold: int = 0, device: str = "cuda"):
    model_name, model_parameters = model_config["model"]["name"], model_config["model"]["parameters"]
    model_paths = model_config["checkpoint"].get("paths", [])
    use_cnn_features = False if model_name in ["rawnet", "rawnet3", "frontend_lcnn", "frontend_specrnet"] else True

    model = models.get_model(
        model_name=model_name, config=model_parameters, device=device,
    )
    # If provided weights, apply corresponding ones (from an appropriate fold)
    weights_path = ""
    # It was our pain!
    if len(model_paths) >= 1:
        assert len(model_paths) == 3 or len(model_paths) == 1, "Pass either 0, 1 or 3 weights path"
        weights_path = model_paths[fold]

        model.load_state_dict(
            torch.load(weights_path)
        )
        LOGGER.info("Loaded weigths on '%s' model, path: %s", model_name, weights_path)
    model = model.to(device)
    model.use_cnn_features = use_cnn_features
    model.weights_path = weights_path

    return model


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

    default_amount = None
    parser.add_argument(
        "--amount",
        "-a",
        help=f"Amount of files to load - useful when debugging (default: {default_amount} - use all).",
        type=int,
        default=default_amount,
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

    parser.add_argument(
        "--no_fold", 
        help="Use no fold version of the dataset", 
        action="store_true"
    )

    return parser.parse_args()


def neptune_init(args, config: Dict, attack_model_config: Optional[Dict]):
    try:
        from neptune.new.integrations.python_logger import NeptuneHandler

        neptune_instance = neptune_utils.get_metric_logger(
            should_log_metrics=config["logging"]["log_metrics"],
            config=config,
            api_token_path="../configs/tokens/neptune_api_token",
        )
        npt_handler = NeptuneHandler(run=neptune_instance)
        LOGGER.addHandler(npt_handler)
    except Exception as e:
        LOGGER.info(e)
        from unittest.mock import MagicMock

        LOGGER.info("Neptune not initialized - using mocked logger!")
        neptune_instance = MagicMock()

    neptune_instance["sys/tags"].add(config["model"]["name"])
    neptune_instance["sys/tags"].add("adversarial_training")
    if args.finetune:
        neptune_instance["sys/tags"].add("adversarial_finetune")
    seed_name = f"seed_{config['data'].get('seed', '?')}"

    neptune_instance["sys/tags"].add(seed_name)
    neptune_instance["source_code/config"].upload(args.config)

    data_frontend = config['data']['cnn_features_setting']['frontend_algorithm']
    if data_frontend:
        # if it is network where we process data before inputting
        neptune_instance["sys/tags"].upload(data_frontend[0])
    else:
        data_frontend = config['model']['parameters'].get('frontend_algorithm', None)
        if data_frontend:
            neptune_instance["sys/tags"].add(data_frontend[0])
        else:
            neptune_instance["sys/tags"].add('raw_audio')

    if attack_model_config is not None:
        neptune_instance["source_code/attack_model_config"].upload(args.attack_model_config)
        neptune_instance["sys/tags"].add(f"attack__{attack_model_config['model']['name']}")

    return neptune_instance


if __name__ == "__main__":
    main(parse_args())
