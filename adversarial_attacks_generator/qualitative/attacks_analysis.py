import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from pathlib import Path

LOGGER = logging.getLogger()


class AttackAnalyser:

    def __init__(self, result_dst):
        self.result_dst = Path(result_dst)
        self.result_dst.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tensor_to_ndarray(
        batch_x,
        batch_x_attacked,
        batch_y,
        batch_preds_label,
        batch_preds,
        batch_preds_noattack_label,
        batch_preds_noattack,
        batch_metadata
    ):
        batch_x = batch_x.cpu().numpy()
        batch_x_attacked = batch_x_attacked.cpu().numpy()

        batch_y = batch_y.cpu().numpy()
        batch_preds_label = batch_preds_label.cpu().numpy()
        batch_preds = batch_preds.cpu().numpy()

        batch_preds_noattack_label = batch_preds_noattack_label.cpu().numpy()
        batch_preds_noattack = batch_preds_noattack.cpu().numpy()

        batch_metadata = list(zip(*batch_metadata))

        return (
            batch_x,
            batch_x_attacked,
            batch_y,
            batch_preds_label,
            batch_preds,
            batch_preds_noattack_label,
            batch_preds_noattack,
            batch_metadata
        )

    @staticmethod
    def sample_diffs(
            batch_x,
            batch_x_attacked,
            batch_y,
            batch_preds_label,
            batch_preds,
            batch_preds_noattack_label,
            batch_preds_noattack,
            batch_data
    ):

        for i in range(len(batch_x)):
            print(i, np.mean(np.abs(batch_x[i] - batch_x_attacked[i])),
                  batch_preds_noattack_label[i] != batch_preds_label[i],
                  "y:", batch_y[i], "y_noadvatk_pred:", batch_preds_noattack_label[i], "y_pred:", batch_preds_label[i],
                  *batch_data[i])

    def save_false_positives(
        self,
        batch_x,
        batch_x_attacked,
        batch_y,
        batch_preds_label,
        batch_preds,
        batch_preds_noattack_label,
        batch_preds_noattack,
        batch_metadata
    ):
        false_positives = np.where(
            (batch_y == np.zeros_like(batch_y))
            & (batch_preds_noattack_label == batch_y)
            & (batch_preds_noattack_label != batch_preds_label)
        )
        # the above operation generates tuple (not sure why)
        false_positives = false_positives[0]
        LOGGER.info("false_positives: {}".format(false_positives))
        self.save_waves(false_positives, batch_x, batch_x_attacked, batch_metadata, "fp")

    def save_false_negatives(
        self,
        batch_x,
        batch_x_attacked,
        batch_y,
        batch_preds_label,
        batch_preds,
        batch_preds_noattack_label,
        batch_preds_noattack,
        batch_metadata
    ):
        false_negatives = np.where(
            (batch_y == np.ones_like(batch_y))
            & (batch_preds_noattack_label == batch_y)
            & (batch_preds_noattack_label != batch_preds_label)
        )
        # the above operation generates tuple (not sure why)
        false_negatives = false_negatives[0]
        LOGGER.info("false_negatives: {}".format(false_negatives))
        self.save_waves(false_negatives, batch_x, batch_x_attacked, batch_metadata, "fn")

    def save_waves(self, false_samples, batch_x, batch_x_attacked, batch_metadata, suffix):
        for i in false_samples:
            src_path = Path(batch_metadata[i][1])
            fold, subset, sec_length = batch_metadata[i][2], batch_metadata[i][3], batch_metadata[i][4]

            if "WaveFake" in str(src_path) or "FakeAVCeleb" in str(src_path):
                src_folder = src_path.parent.relative_to(src_path.parents[1])
                file_name = f"{src_folder}_{src_path.stem}"
            else:
                file_name = src_path.stem

            file_name = f"{file_name}_{subset}_fold{fold}_{sec_length:.2f}sec"

            wavfile.write(
                filename=self.result_dst / f"{file_name}_{suffix}_original.wav",
                rate=16_000,
                data=batch_x[i]
            )

            wavfile.write(
                filename=self.result_dst / f"{file_name}_{suffix}_attacked.wav",
                rate=16_000,
                data=batch_x_attacked[i]
            )

    def analyse(self, *args, **kwargs):
        args = self.tensor_to_ndarray(*args, **kwargs)
        self.sample_diffs(*args)
        self.save_false_positives(*args)
        self.save_false_negatives(*args)
        # self.save_plot_false_positives(*args)

