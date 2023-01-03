"""Utility file for src toolkit."""
import logging
import os
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from torch import nn

from src.models import models


LOGGER = logging.getLogger(__name__)


def find_wav_files(path_to_dir: Union[Path, str]) -> Optional[List[Path]]:
    """Find all wav files in the directory and its subtree.

    Args:
        path_to_dir: Path top directory.
    Returns:
        List containing Path objects or None (nothing found).
    """
    paths = list(sorted(Path(path_to_dir).glob("**/*.wav")))

    if len(paths) == 0:
        return None
    return paths


def set_seed(seed: int):
    """Fix PRNG seed for reproducable experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_model(model_config, device: str = "cuda"):
    model_name, model_parameters = model_config["model"]["name"], model_config["model"]["parameters"]
    model_path = model_config["checkpoint"].get("path", "")

    model = models.get_model(
        model_name=model_name, config=model_parameters, device=device,
    )
    # If provided weights, apply corresponding ones (from an appropriate fold)
    if model_path:
        try:
            model.load_state_dict(
                torch.load(model_path)
            )
        except RuntimeError:
            model = nn.DataParallel(model)
            model.load_state_dict(
                torch.load(model_path)
            )
            model = model.module

        LOGGER.info("Loaded weigths on '%s' model, path: %s", model_name, model_path)
    model = model.to(device)
    model.weights_path = model_path

    return model