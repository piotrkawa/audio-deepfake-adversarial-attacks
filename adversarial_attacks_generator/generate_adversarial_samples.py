import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import tqdm
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from adversarial_attacks_generator import utils
from adversarial_attacks_generator.attacks import AttackEnum
from adversarial_attacks_generator.qualitative.attacks_analysis import \
    AttackAnalyser
from src.datasets.attack_agnostic_dataset import (AttackAgnosticDataset,
                                                  NoFoldDataset)
from src.metrics import calculate_eer
from src.models import models
from src.utils import set_seed

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)

Path("logs").mkdir(exist_ok=True)
fh = logging.FileHandler(f"logs/{datetime.now()}.log")
fh.setFormatter(formatter)

LOGGER.addHandler(sh)
LOGGER.addHandler(fh)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # If assigned as None, then it won't be taken into account
    ASVSPOOF_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/ASVspoof2021/DF"
    WAVEFAKE_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/WaveFake"
    FAKEAVCELEB_DATASET_PATH = "/home/adminuser/storage/datasets/deep_fakes/FakeAVCeleb/FakeAVCeleb_v1.2"

    parser.add_argument(
        "--asv_path", type=str, default=ASVSPOOF_DATASET_PATH
    )
    parser.add_argument(
        "--wavefake_path", type=str, default=WAVEFAKE_DATASET_PATH
    )
    parser.add_argument(
        "--celeb_path", type=str, default=FAKEAVCELEB_DATASET_PATH
    )

    parser.add_argument(
        "--attack",
        help="Model config file path",
        type=str,
        default=AttackEnum.NO_ATTACK.name,
        choices=[e.name for e in AttackEnum],
    )

    parser.add_argument(
        "--attack_model_config",
        help="Model config file path",
        type=str,
        default=None
    )

    default_model_config = "configs/lcnn.yaml"
    parser.add_argument(
        "--config",
        help="Model config file path",
        type=str,
        default=default_model_config
    )

    default_amount = None
    parser.add_argument(
        "--amount", "-a",
        help=f"Amount of files to load from each directory (default: {default_amount} - use all).",
        type=int,
        default=default_amount
    )

    parser.add_argument(
        "--qual",
        help="Generate qualitative results",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--no_fold", 
        help="Use no fold version of the dataset",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--raw_from_dataset",
        help="Return raw sample from the dataset",
        default=False,
        action="store_true"
    )

    return parser.parse_args()


def main(args):
    print(args)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if args.attack_model_config is not None:
        with open(args.attack_model_config, "r") as f:
            attack_model_config = yaml.safe_load(f)
    else:
        attack_model_config = None

    with open(args.config, "r") as f:  # we test this
        config = yaml.safe_load(f)

    seed = config["data"].get("seed", 42)
    # fix all seeds - this should not actually change anything
    set_seed(seed)

    attack_method, attack_params = AttackEnum[args.attack].value

    if args.qual:
        results_folder = f"attack_{args.attack}_{Path(args.attack_model_config).stem}_on_{Path(args.config).stem}"
        attack_analyser = AttackAnalyser(Path("qualitative_results") / results_folder)
        on_attack_end_callback = attack_analyser.analyse
    else:
        on_attack_end_callback = None

    generate_attacks(
        datasets_paths=[args.asv_path, args.wavefake_path, args.celeb_path],
        model_config=config,
        attack_model_config=attack_model_config,
        attack_method=attack_method,
        attack_params=attack_params,
        amount_to_use=args.amount,
        device=device,
        on_attack_end_callback=on_attack_end_callback,
        no_fold=args.no_fold,
        raw_sample_from_dataset=args.raw_from_dataset
    )

def load_model(model_config, fold: int = 0, device: str = "cuda"):
    model_name, model_parameters = model_config["model"]["name"], model_config["model"]["parameters"]
    model_paths = model_config["checkpoint"].get("paths", [])

    model = models.get_model(
        model_name=model_name, config=model_parameters, device=device,
    )
    # If provided weights, apply corresponding ones (from an appropriate fold)
    weights_path = ""
    if len(model_paths) >= 1:
        assert len(model_paths) == 3 or len(model_paths) == 1, "Pass either 0, 1 or 3 weights path"
        weights_path = model_paths[fold]

        try:
            model.load_state_dict(
                torch.load(weights_path)
            )
        except RuntimeError:
            model = nn.DataParallel(model)
            model.load_state_dict(
                torch.load(weights_path)
            )
            model = model.module

        LOGGER.info("Loaded weigths on '%s' model, path: %s", model_name, weights_path)
    model = model.to(device)
    model.weights_path = weights_path

    return model


def generate_attacks(
    datasets_paths: List[Union[Path, str]],
    model_config: Dict,
    device: str,
    attack_model_config: Optional[Dict] = None,
    attack_method: Optional[Any] = None,
    attack_params: Dict = {},
    amount_to_use: Optional[int] = None,
    batch_size: int = 64,
    on_attack_end_callback: Optional[Callable] = None,
    raw_sample_from_dataset: bool = False,
    no_fold: bool = False,
):
    LOGGER.info("Loading data...")

    folds = [0, 1, 2] if not no_fold else [-1]

    for fold_no, fold in enumerate(tqdm.tqdm(folds)):
        LOGGER.info(f"Test Fold [{fold_no + 1}/{len(folds)}]")
        # Load model architecture
        model = load_model(model_config, fold, device)
        model = nn.DataParallel(model)

        if attack_model_config is not None and attack_method is not None:
            attack_model = load_model(attack_model_config, fold, device)
            attack_model = nn.DataParallel(attack_model)

            atk = attack_method(attack_model, **attack_params)
            atk.set_training_mode(model_training=True, batchnorm_training=False)

        else:
            attack_model = None
            atk = None

        logging_prefix = f"fold_{fold}"

        data_val = get_dataset(
            datasets_paths=datasets_paths,
            fold=fold,
            amount_to_use=amount_to_use,
            raw_sample_from_dataset=raw_sample_from_dataset
        )

        LOGGER.info(
            f"Testing '{model.module.__class__.__name__}' model, "
            f"weights path: '{model.module.weights_path}', "
            f"on {len(data_val)} audio files."
        )

        if attack_model is not None:
            LOGGER.info(
                f"Attack using '{attack_model.module.__class__.__name__}' model "
                f"and '{atk.__class__.__name__}' method ({attack_params}), "
                f"weights path: '{attack_model.module.weights_path}'"
            )
        else:
            LOGGER.info("No attack applied")

        test_loader = DataLoader(
            data_val,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=3,
        )

        num_correct = 0.0
        num_total = 0.0
        y_pred = []
        y = []
        y_pred_label = []

        for i, (batch_x, batch_sr, batch_y, batch_metadata) in tqdm.tqdm(enumerate(test_loader), desc="Batches"):
            model.eval()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            num_total += batch_x.size(0)

            if attack_model is not None:
                batch_x_attacked, mn, mx = utils.to_minmax(batch_x)
                batch_x_attacked = atk(batch_x_attacked, batch_y)
                batch_x_attacked = utils.revert_minmax(batch_x_attacked, mn, mx)
            else:
                batch_x_attacked = torch.clone(batch_x)

            batch_x_noproc = torch.clone(batch_x)
            batch_x_attacked_noproc = torch.clone(batch_x_attacked)

            with torch.no_grad():
                # here we run preprocessing with defaults parameters WAVE_FAKE_CUT, WAVE_FAKE_TRIM, WAVE_FAKE_SR, etc.
                if raw_sample_from_dataset:
                    batch_x_attacked, _ = AttackAgnosticDataset.wavefake_preprocessing_on_batch(
                        batch_x_attacked,
                        batch_sr,
                    )

                batch_preds = model(batch_x_attacked).squeeze(1).detach()
                batch_preds = torch.sigmoid(batch_preds)
                batch_preds_label = (batch_preds + .5).int()

                if on_attack_end_callback is not None:
                    if raw_sample_from_dataset:
                        batch_x, _ = AttackAgnosticDataset.wavefake_preprocessing_on_batch(
                            batch_x,
                            batch_sr,
                        )
                    batch_preds_noattack = model(batch_x).squeeze(1).detach()
                    batch_preds_noattack = torch.sigmoid(batch_preds_noattack)
                    batch_preds_noattack_label = (batch_preds_noattack + .5).int()

                    on_attack_end_callback(
                        batch_x=batch_x_noproc,
                        batch_x_attacked=batch_x_attacked_noproc,
                        batch_y=batch_y,
                        batch_preds_label=batch_preds_label,
                        batch_preds=batch_preds,
                        batch_preds_noattack_label=batch_preds_noattack_label,
                        batch_preds_noattack=batch_preds_noattack,
                        batch_metadata=batch_metadata,
                    )

            num_correct += (batch_preds_label == batch_y.int()).sum(dim=0).item()

            y_pred.append(batch_preds.cpu().numpy()) # torch.concat([y_pred, batch_pred], dim=0)
            y_pred_label.append(batch_preds_label.cpu().numpy()) # torch.concat([y_pred_label, batch_pred_label], dim=0)
            y.append(batch_y.cpu().numpy()) # torch.concat([y, batch_y], dim=0)

        eval_accuracy = (num_correct / num_total) * 100

        y_pred = np.concatenate(y_pred, axis=0)
        y_pred_label = np.concatenate(y_pred_label, axis=0)
        y = np.concatenate(y, axis=0)
        precision, recall, f1_score, support = precision_recall_fscore_support(
            y,
            y_pred_label,
            average="binary",
            beta=1.0
        )
        auc_score = roc_auc_score(y_true=y, y_score=y_pred)

        # For EER flip values, following original evaluation implementation
        y_for_eer = 1 - y

        thresh, eer, fpr, tpr = calculate_eer(
            y=y_for_eer,
            y_score=y_pred,
        )

        eer_label = f"adv_eval/{logging_prefix}__eer"
        accuracy_label = f"adv_eval/{logging_prefix}__accuracy"
        precision_label = f"adv_eval/{logging_prefix}__precision"
        recall_label = f"adv_eval/{logging_prefix}__recall"
        f1_label = f"adv_eval/{logging_prefix}__f1_score"
        auc_label = f"adv_eval/{logging_prefix}__auc"

        logger[eer_label].log(eer)
        logger[accuracy_label].log(eval_accuracy)
        logger[precision_label].log(precision)
        logger[recall_label].log(recall)
        logger[f1_label].log(f1_score)
        logger[auc_label].log(auc_score)

        LOGGER.info(
            f"{eer_label}: {eer:.4f}, {accuracy_label}: {eval_accuracy:.4f}, {precision_label}: {precision:.4f}, "
            f"{recall_label}: {recall:.4f}, {f1_label}: {f1_score:.4f}, {auc_label}: {auc_score:.4f}"
        )


def get_dataset(
    datasets_paths: List[Union[Path, str]],
    fold: int,
    amount_to_use: Optional[int],
    raw_sample_from_dataset: bool = False
) -> Union[AttackAgnosticDataset, NoFoldDataset]:
    if fold == -1: # no_fold setting
        data_val = NoFoldDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="val",
            reduced_number=10_000,
            return_label=True,
            return_meta=True,
            return_raw=raw_sample_from_dataset
        )
    else:
        data_val = AttackAgnosticDataset(
            asvspoof_path=datasets_paths[0],
            wavefake_path=datasets_paths[1],
            fakeavceleb_path=datasets_paths[2],
            fold_num=fold,
            fold_subset="val",
            reduced_number=amount_to_use,
            return_label=True,
            return_meta=True,
            return_raw=raw_sample_from_dataset
        )
    return data_val


if __name__ == "__main__":
    main(parse_arguments())
