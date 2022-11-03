from typing import Dict

def feature_kwargs(lfcc: bool) -> dict:
    """Settings for preprocessing.
    """
    if lfcc:
        return {
            "n_lin": 20,
            "n_lfcc": 20,
            "log_lf": True,
            "speckwargs": {
                "n_fft": 512,
            },
        }
    else: 
        return {
            "n_mfcc": 20,
            "log_mels": True,
            "melkwargs": {
                "n_mels": 20,
                "n_fft": 512,
            }
        }


RAW_NET_CONFIG = {
    "nb_samp": 64600,
    "first_conv": 1024,   # no. of filter coefficients
    "in_channels": 1,  # no. of filters channel in residual blocks
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 1,
}

# TODO(PK): rename this file
def get_specrnet_config(input_channels: int) -> Dict:
    return {
        "filts": [input_channels, [input_channels, 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 1,
    }

