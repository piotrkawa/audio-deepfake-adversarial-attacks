from copy import deepcopy
from typing import Dict

from dfadetect.models import lcnn, mesonet, raw_net2, rawnet3, specrnet, xception
from experiment_config import RAW_NET_CONFIG, get_specrnet_config


def get_model(model_name: str, config: Dict, device: str):
    if model_name == "rawnet":
        return raw_net2.RawNet(deepcopy(RAW_NET_CONFIG), device=device)
    elif model_name == "mesonet_inception":
        return mesonet.MesoInception4(num_classes=1, **config)
    elif model_name == "lcnn":
        return lcnn.LCNN(**config)
    elif model_name == "rawnet3":
        return rawnet3.prepare_model()
    elif model_name == "frontend_lcnn":
        return lcnn.FrontendLCNN(device=device, **config)
    elif model_name == "frontend_specrnet":
        return specrnet.FrontendSpecRNet(
            get_specrnet_config(config.get("input_channels", 1)),
            device=device,
            **config,
        )
    elif model_name == "specrnet":
        return specrnet.SpecRNet(
            get_specrnet_config(config.get("input_channels", 1)), device=device
        )
    elif model_name == "xception":
        return xception.xception(num_classes=1, pretrained=None, **config)
    else:
        raise ValueError(f"Model '{model_name}' not supported")
