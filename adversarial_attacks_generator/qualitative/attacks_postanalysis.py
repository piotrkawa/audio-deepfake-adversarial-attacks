import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from mel_cepstral_distance import get_metrics_wavs


print(plt.style.available)
plt.style.use('ggplot')

LOGGER = logging.getLogger()

class AttackPostAnalyser:

    def __init__(self, result_dst):
        self.result_dst = Path(result_dst)
        self.result_dst.mkdir(parents=True, exist_ok=True)

    def save_plot_false_positives(
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
        print("false_positives:", false_positives)

        self.save_plots(false_positives, batch_x, batch_x_attacked, batch_metadata, "fp")

    def save_plots(self, false_samples, batch_x, batch_x_attacked, batch_metadata, suffix):
        for i in false_samples:
            src_path = Path(batch_metadata[i][1])
            fold, subset, sec_length = batch_metadata[i][2], batch_metadata[i][3], batch_metadata[i][4]

            if "WaveFake" in str(src_path) or "FakeAVCeleb" in str(src_path):
                src_folder = src_path.parent.relative_to(src_path.parents[1])
                file_name = f"{src_folder}_{src_path.stem}"
            else:
                file_name = src_path.stem

            file_name = f"{file_name}_{subset}_fold{fold}_{sec_length:.2f}sec_{suffix}"

            self.save_plot(file_name, batch_x[i], batch_x_attacked[i], (1100, 1200))

    def save_plot(self, file_name, xo, xa, rang):
        plt.rcParams["figure.figsize"] = [12.0, 4.0]
        s, e = rang
        plt.plot(xo[s:e], "--", color="steelblue")
        plt.plot(xa[s:e], "-", color="yellow")
        plt.plot(xo[s:e] - xa[s:e], color="lightcoral")
        plt.savefig(self.result_dst / f"{file_name}_plot.png")
        # plt.show()
        plt.clf()

    def read_waves_and_plot(self, path=None):
        if path is None:
            path = self.result_dst

        wav_paths = list(Path(path).glob("**/*.wav"))
        wav_paths = [w for w in wav_paths if "original" in str(w)]

        for path in wav_paths:
            stem = "_".join(path.stem.split("_")[:-1])
            att_path = path.parent / f"{stem}_attacked.wav"

            _, org_wav = wavfile.read(path)
            _, att_wav = wavfile.read(att_path)

            self.save_plot(stem, org_wav, att_wav, (1100, 1300))

    def calc_mock(self, file_name, xo, xa):
        return {
            "name": [file_name],
            "mock_original": [1.0],
            "mock_attacked": [1.0]
        }

    def read_waves_and_calc_metrics(self, path=None):
        if path is None:
            path = self.result_dst

        wav_paths = list(Path(path).glob("**/*.wav"))
        wav_paths = [w for w in wav_paths if "original" in str(w)]

        results = pd.DataFrame()
        mcd_results = []
        import tqdm
        for path in tqdm.tqdm(wav_paths):
            stem = "_".join(path.stem.split("_")[:-1])
            att_path = path.parent / f"{stem}_attacked.wav"

            _, org_wav = wavfile.read(path)
            _, att_wav = wavfile.read(att_path)
            try:
                distance, penalty, frames = get_metrics_wavs(Path(path), Path(att_path))
            except:
                LOGGER.info("ERROR")
                continue
            mcd_results.append(distance)

            r = self.calc_mock(stem, org_wav, att_wav)
            r_df = pd.DataFrame(r)

            results = pd.concat([results, r_df], ignore_index=True)

        results = results.reset_index(drop=True)
        results.to_csv(path.parent / "metrics.csv")

        mcd_results = np.array(mcd_results)
        mcd_csv_data = np.array([mcd_results.mean(), mcd_results.std(), mcd_results.min(), mcd_results.max()])
        mcd_csv_data = np.expand_dims(mcd_csv_data, 0)
        mcd_csv_data = pd.DataFrame(mcd_csv_data, columns=["mean", "std", "min", "max"])
        mcd_csv_data.to_csv(path.parent / "mcd_metrics.csv")
        LOGGER.info("MCD: {}, {}, {}, {}".format(mcd_results.mean(), mcd_results.std(), mcd_results.min(), mcd_results.max()))
        return results


if __name__ == "__main__":

    # results_dir = "/home/marcin/Projects/deepfake/aad-plus-plus/" \
    #               "adversarial_attacks_generator/qualitative_results/" \
    #               "attack_CW_frontend_lcnn_on_frontend_specrnet"
    # results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_FGSM_frontend_lcnn_on_frontend_lcnn"
    results = []

    results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_FGSM_eps00075_frontend_lcnn_on_frontend_lcnn"  # MCD:  3.460303410403729 1.6950860163473498 0.5395627846911826 11.134754686011629
    results.append(results_dir)
    results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_FGSM_eps001_frontend_lcnn_on_frontend_lcnn"  # MCD:  4.182125240838386 1.7725816776546697 0.7692531207446074 11.90145505273958
    results.append(results_dir)

    results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_PGDL2_frontend_lcnn_on_frontend_lcnn"  # MCD:  2.4450682744261125 1.3242916554171738 0.4071250629830576 12.013154299247496
    results.append(results_dir)
    results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_PGDL2_eps15_frontend_lcnn_on_frontend_lcnn"  # MCD:  3.6379317362884627 1.6574908397560482 0.6268950041181226 13.49373124542276
    results.append(results_dir)
    results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_PGDL2_eps20_frontend_lcnn_on_frontend_lcnn"  # MCD:  4.500306913392032 1.960941567271789 0.7657868217829195 15.354959733282774
    results.append(results_dir)
    
    results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_FAB_frontend_lcnn_on_frontend_lcnn"  # MCD:  2.3111574352533606 1.362631944466171 0.048595166431498214 10.103028028734432
    results.append(results_dir)
    results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_FAB_eta20_frontend_lcnn_on_frontend_lcnn"  # MCD:  3.5086876553766384 1.8392795661212549 0.027590385624640758 11.174434147692214
    results.append(results_dir)
    results_dir = "/home/adminuser/pietrek/aad-plus-plus/adversarial_attacks_generator/qualitative_results/attack_FAB_eta30_frontend_lcnn_on_frontend_lcnn"  # MCD:  4.469023378989573 2.2714712339830743 0.04157005919294463 12.777830928895568
    results.append(results_dir)

    for res in results:
        analyser = AttackPostAnalyser(result_dst=res)
        # analyser.read_waves_and_plot()
        r = analyser.read_waves_and_calc_metrics()
        LOGGER.info(np.mean(r["mock_original"]))
