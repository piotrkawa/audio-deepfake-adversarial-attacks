# Defense against Adversarial Attacks on Audio DeepFake Detection

The following repository contains code for our paper called ["Defense against Adversarial Attacks on Audio DeepFake Detection"](https://arxiv.org/abs/2212.14597).


We base our codebase on [Attack Agnostic Dataset repo](https://github.com/piotrkawa/attack-agnostic-dataset).

## Demo samples
You can find demo samples [here](https://piotrkawa.github.io/papers/adversarial_attacks.html)

## Before you start


### Datasets

Download appropriate datasets:

* [ASVspoof2021 DF subset](https://zenodo.org/record/4835108),
* [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb#access-request-form),
* [WaveFake](https://zenodo.org/record/5642694) (along with JSUT and LJSpeech).



### Dependencies
Install required dependencies using: 
```bash
pip install -r requirements.txt
```

### Configs

Both training and evaluation scripts are configured with the use of CLI and `.yaml` configuration files. File defines processing applied to raw audio files, as well as used architecture. An example config of LCNN architecture with LFCC frontend looks as follows:
```yaml
data:
  seed: 42

checkpoint: 
  # This part is used only in evaluation 
  path: "trained_models/aad__lcnn/ckpt.pth",

model:
  name: "lcnn"  # {"lcnn", "specrnet", "rawnet3"}
  parameters:
    input_channels: 1
  optimizer:
    lr: 0.0001
```

Other example configs are available under `configs/training/`

##  Train models 


To train models use `train_models.py`. 


```
usage: train_models.py [-h] [--asv_path ASV_PATH] [--wavefake_path WAVEFAKE_PATH] [--celeb_path CELEB_PATH] [--config CONFIG] [--amount AMOUNT] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--ckpt CKPT] [--cpu]

optional arguments:
  -h, --help            show this help message and exit
  --asv_path ASV_PATH   Path to ASVspoof2021 dataset directory
  --wavefake_path WAVEFAKE_PATH
                        Path to WaveFake dataset directory
  --celeb_path CELEB_PATH
                        Path to FakeAVCeleb dataset directory
  --config CONFIG       Model config file path (default: config.yaml)
  --train_amount TRAIN_AMOUNT, -a TRAIN_AMOUNT
                        Amount of files to load for training.
  --test_amount TEST_AMOUNT, -ta TEST_AMOUNT
                        Amount of files to load for testing.
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch size (default: 128).
  --epochs EPOCHS, -e EPOCHS
                        Epochs (default: 5).
  --ckpt CKPT           Checkpoint directory (default: trained_models).
  --cpu, -c             Force using cpu?
```

## Evaluate models


Once your models are trained you can evalaute them using `evaluate_models.py`.

**Before you start:** add checkpoint paths to the config used in training process.



```
usage: evaluate_models.py [-h] [--asv_path ASV_PATH] [--wavefake_path WAVEFAKE_PATH] [--celeb_path CELEB_PATH] [--config CONFIG] [--amount AMOUNT] [--cpu] 

optional arguments:
  -h, --help            show this help message and exit
  --asv_path ASV_PATH
  --wavefake_path WAVEFAKE_PATH
  --celeb_path CELEB_PATH
  --config CONFIG       Model config file path (default: config.yaml)
  --amount AMOUNT, -a AMOUNT
                        Amount of files to load from each directory (default: None - use all).
  --cpu, -c             Force using cpu
```
e.g. to evaluate LCNN network add appropriate checkpoint paths to config and then use:
```
python evaluate_models.py --config configs/training/lcnn.yaml --asv_path ../datasets/ASVspoof2021/DF --wavefake_path ../datasets/WaveFake --celeb_path ../datasets/FakeAVCeleb/FakeAVCeleb_v1.2
```

## Adversarial Evaluation



Attack LCNN network using white-box setting with FGSM attack:
```bash
python generate_adversarial_samples.py --attack FGSM --config configs/frontend_lcnn.yaml --attack_model_config configs/frontend_lcnn.yaml --raw_from_dataset
```

Attack LCNN network using transferability setting with FGSM attack based on RawNet3:
```bash
python generate_adversarial_samples.py --attack FGSM --config configs/frontend_lcnn.yaml --attack_model_config configs/rawnet3.yaml --raw_from_dataset
```

## Adversarial Training


Finetune LCNN model for 10 epochs using a `` strategy:
```bash
python train_models_with_adversarial_attacks.py --config {config} --epochs 10 --adv_training_strategy {args.adv_training_strategy} --attack_model_config {attack_model_config} --finetune
```

## Acknowledgments

Apart from the dependencies mentioned in Attack Agnostic Dataset repository we also include: 
* [RawNet3 implementation](https://github.com/Jungjee/RawNet), 
* [Adversarial-Attacks-PyTorch repository](https://github.com/Harry24k/adversarial-attacks-pytorch) - please note that we slightly modified it. The `adversarial_attacks` source code placed in our repository handles single value outputs and wave inputs, e.g., we create a two element vector based on a single value output as follows:
```python
outputs = self.model(images)
outputs = torch.cat([-outputs, outputs], dim=1)
```
Note that only selected adversarial attacks are handled: FGSM, FAB, PGD, PGDL2, OnePixel and CW.

## Citation

If you use this code in your research please use the following citation:

```
@misc{kawa2022defense,
    title={Defense Against Adversarial Attacks on Audio DeepFake Detection},
    author={Piotr Kawa and Marcin Plata and Piotr Syga},
    year={2022},
    eprint={2212.14597},
    archivePrefix={arXiv},
    primaryClass={cs.SD}
}
```

