from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from dfadetect.datasets import TransformDataset
from dfadetect.models.gaussian_mixture_model import GMMBase, classify_dataset


def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y, -y_score)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer, fpr, tpr


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    training_dataset_name: str,
    fake_dataset_name: str,
    path: str,
    lw: int = 2,
    save: bool = False,
) -> matplotlib.figure.Figure:
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title(
    # f'Train: {training_dataset_name}\nEvaluated on {fake_dataset_name}')
    ax.legend(loc="lower right")

    fig.tight_layout()
    if save:
        fig.savefig(f"{path}.pdf")
    plt.close(fig)
    return fig


def calculate_eer_for_gmm(
    real_model: GMMBase,
    fake_model: GMMBase,
    real_dataset_test: TransformDataset,
    fake_dataset_test: TransformDataset,
    training_dataset_name: str,
    fake_dataset_name: str,
    plot_dir_path: str,
    device: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    real_scores = classify_dataset(
        real_model, fake_model, real_dataset_test, device
    ).numpy()

    fake_scores = classify_dataset(
        real_model, fake_model, fake_dataset_test, device
    ).numpy()

    # JSUT fake samples are fewer available
    length = min(len(real_scores), len(fake_scores))
    real_scores = real_scores[:length]
    fake_scores = fake_scores[:length]

    labels = np.concatenate(
        (
            np.zeros(real_scores.shape, dtype=np.int32),
            np.ones(fake_scores.shape, dtype=np.int32),
        )
    )

    thresh, eer, fpr, tpr = calculate_eer(
        y=np.array(labels, dtype=np.int32),
        y_score=np.concatenate((real_scores, fake_scores)),
    )

    fig_path = f"{plot_dir_path}/{training_dataset_name.replace('.', '_').replace('/', '_')}_{fake_dataset_name.replace('.', '_').replace('/', '_')}"
    plot_roc(fpr, tpr, training_dataset_name, fake_dataset_name, fig_path)

    return eer, thresh, fpr, tpr
