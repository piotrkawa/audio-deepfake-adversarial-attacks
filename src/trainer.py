"""A generic training wrapper."""
import functools
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from adversarial_attacks_generator import utils
from adversarial_attacks_generator.aa_types import AttackEnum


LOGGER = logging.getLogger(__name__)


def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
    epoch: Optional[int] = None
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    if epoch is not None:
        epoch_str = f"_{epoch:02d}"
    else:
        epoch_str = ""
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt{epoch_str}.pth")
    LOGGER.info(f"Training model saved under: {full_model_dir}/ckpt{epoch}.pth")


class Trainer():
    """This is a lightweight wrapper for training models with gradient descent.

    Its main function is to store information about the training process.

    Args:
        epochs (int): The amount of training epochs.
        batch_size (int): Amount of audio files to use in one batch.
        device (str): The device to train on (Default 'cpu').
        batch_size (int): The amount of audio files to consider in one batch (Default: 32).
        optimizer_fn (Callable): Function for constructing the optimzer.
        optimizer_kwargs (dict): Kwargs for the optimzer.
    """

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        use_scheduler: bool = False,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []
        self.use_scheduler = use_scheduler


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss


class GDTrainer(Trainer):

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            lengths = [train_len, test_len]
            train, test = torch.utils.data.random_split(dataset, lengths)

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss

        if self.use_scheduler:
            batches_per_epoch = len(train_loader) * 2  # every 2nd epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optim,
                T_0=batches_per_epoch,
                T_mult=1,
                eta_min=5e-6,
                # verbose=True,
            )
        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                if i % 50 == 0:
                    lr = scheduler.get_last_lr()[0] if self.use_scheduler else self.optimizer_kwargs["lr"]

                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                batch_out, batch_loss = forward_and_loss_fn(model, criterion, batch_x, batch_y, use_cuda=use_cuda)
                batch_pred = (torch.sigmoid(batch_out) + .5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

                running_loss += (batch_loss.item() * batch_size)

                if i % 100 == 0:
                    LOGGER.info(
                         f"[{epoch:04d}][{i:05d}]: {running_loss / num_total} {num_correct/num_total*100}")

                optim.zero_grad()
                batch_loss.backward()
                optim.step()
                if self.use_scheduler:
                    scheduler.step()

            running_loss /= num_total
            train_accuracy = (num_correct / num_total) * 100

            LOGGER.info(f"Epoch [{epoch+1}/{self.epochs}]: train/loss: {running_loss}, train/accuracy: {train_accuracy}")

            test_running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            model.eval()
            eer_val = 0

            for batch_x, _, batch_y in test_loader:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                with torch.no_grad():
                    batch_pred = model(batch_x)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                batch_loss = criterion(batch_pred, batch_y)

                test_running_loss += (batch_loss.item() * batch_size)

                batch_pred = torch.sigmoid(batch_pred)
                batch_pred_label = (batch_pred + .5).int()
                num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

            if num_total == 0:
                num_total = 1

            test_running_loss /= num_total
            test_acc = 100 * (num_correct / num_total)
            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"test/loss: {test_running_loss}, "
                f"test/accuracy: {test_acc}, "
                f"test/eer: {eer_val}"
            )

            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())

            LOGGER.info(
                f"Epoch [{epoch:04d}]: loss: {running_loss}, train acc: {train_accuracy}, test_acc: {test_acc}")

        model.load_state_dict(best_model)
        return model


class AdversarialGDTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(AdversarialGDTrainer, self).__init__(*args, **kwargs)

        self.attacks = None

    @staticmethod
    def multi_f1_score(results):
        s = sum(results)
        m = functools.reduce(lambda x, y: x * y, results)
        return len(results) * m / s

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        attack_model: torch.nn.Module,
        adversarial_attacks: List[str],
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        model_dir: Optional[str] = None,
        save_model_name: Optional[str] = None
    ):

        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            lengths = [train_len, test_len]
            train, test = torch.utils.data.random_split(dataset, lengths)

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        best_model = None
        best_acc = 0

        LOGGER.info(f"Starting adversarial training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss

        if self.use_scheduler:
            LOGGER.info("Using optimizer scheduler!")
            batches_per_epoch = len(train_loader)  # number of batches
            eta_min = self.optimizer_kwargs.get("eta_min", 5e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optim,
                T_0=batches_per_epoch,
                T_mult=1,
                eta_min=eta_min,
                # verbose=True,
            )
        use_cuda = self.device != "cpu"

        # Initialize attacks
        self.init_adv_attacks(attack_model, adversarial_attacks)

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            # Train
            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size

                # Prepare input
                batch_x = batch_x.to(self.device)

                # Apply adversarial attack
                batch_x = self.apply_adv_attack(batch_x, batch_y)
                batch_x = batch_x.detach()

                # Prepare ground truth
                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                batch_out, batch_loss = forward_and_loss_fn(
                    model=model,
                    criterion=criterion,
                    batch_x=batch_x,
                    batch_y=batch_y,
                    use_cuda=use_cuda,
                )
                batch_pred = (torch.sigmoid(batch_out) + .5).int()
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

                running_loss += (batch_loss.item() * batch_size)

                if i % 100 == 0:
                    LOGGER.info(
                         f"[{epoch:04d}][{i:05d}]: {running_loss / num_total} {num_correct/num_total*100}")

                optim.zero_grad()
                batch_loss.backward()
                optim.step()
                if self.use_scheduler:
                    scheduler.step()

                self.update_adv_attack(batch_loss.detach().cpu().numpy(), batch_pred, iter=i, epoch=epoch)

            running_loss /= num_total
            train_accuracy = (num_correct/num_total) * 100

            LOGGER.info(f"Epoch [{epoch+1}/{self.epochs}]: train/loss: {running_loss}, "
                        f"train/accuracy: {train_accuracy}")

            # Validation
            test_running_loss, test_acc, eer_val = self.validation_epoch(
                model=model,
                criterion=criterion,
                test_loader=test_loader,
                attack=None,
            )
            test_acc_results = [test_acc / 100]

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: test/loss: {test_running_loss}, "
                f"test/accuracy: {test_acc}, test/eer: {eer_val}"
            )

            # Adversary validation
            for (attack_name, attack_method) in self.attacks:
                # reinitialize to get the same samples
                # TODO: make sure that the data is the same
                test_loader = DataLoader(
                    test,
                    batch_size=self.batch_size,
                    drop_last=True,
                    shuffle=True,
                    num_workers=6,
                )

                adv_test_running_loss, adv_test_acc, adv_eer_val = self.validation_epoch(
                    model=model,
                    criterion=criterion,
                    test_loader=test_loader,
                    attack=attack_method,
                )
                test_acc_results.append(adv_test_acc / 100)

                LOGGER.info(
                    f"Epoch [{epoch+1}/{self.epochs}]: "
                    f"adv_test/{attack_name}__loss: {adv_test_running_loss},"
                    f" adv_test/{attack_name}__accuracy: {adv_test_acc},"
                    f" adv_test/{attack_name}__eer: {adv_eer_val}."
                )

            LOGGER.info(
                f"[{epoch:04d}]: loss {running_loss}, train acc: {train_accuracy}, test_acc: {test_acc}"
            )

            test_acc = self.multi_f1_score(test_acc_results)
            LOGGER.info(f"[{epoch:04d}]: multi_f1_score: {test_acc}")

            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())
                LOGGER.info(f"[{epoch:04d}]: update best model")

            if model_dir is not None:
                save_model(
                    model=model,
                    model_dir=model_dir,
                    name=save_model_name,
                    epoch=epoch
                )

        model.load_state_dict(best_model)
        return model

    def validation_epoch(
        self,
        model,
        test_loader,
        criterion,
        attack: Optional[Callable],
    ):
        model.eval()
        test_running_loss = 0.0
        num_correct = 0.0
        num_total = 0.0
        batch_no = 0
        eer_val = 0  # TODO: revert EER support

        for batch_x, _, batch_y in test_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(self.device)

            if attack:
                batch_x, mn, mx = utils.to_minmax(batch_x)
                batch_x = attack(batch_x, batch_y)
                batch_x = utils.revert_minmax(batch_x, mn, mx)

            batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

            with torch.no_grad():
                batch_pred = model(batch_x)

            batch_loss = criterion(batch_pred, batch_y)

            test_running_loss += (batch_loss.item() * batch_size)

            batch_pred = torch.sigmoid(batch_pred)
            batch_pred_label = (batch_pred + .5).int()
            num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()
            batch_no += 1

        if num_total == 0:
            num_total = 1

        test_running_loss /= num_total
        test_acc = 100 * (num_correct / num_total)

        return test_running_loss, test_acc, eer_val

    def init_adv_attacks(self, attack_model, adversarial_attacks):
        self.attacks = []
        # Initialize attacks
        for attack_method_name in adversarial_attacks:
            attack_method, attack_params = AttackEnum[attack_method_name].value
            atk = attack_method(attack_model, **attack_params)
            atk.set_training_mode(model_training=True, batchnorm_training=False)
            attack_info = (attack_method_name, atk)  # name, method
            self.attacks.append(attack_info)
        LOGGER.info(f"Adversarial attacks: {adversarial_attacks}")

        return self.attacks

    def apply_adv_attack(self, batch_x, batch_y,):
        if random.random() > 1 / (len(self.attacks) + 1):
            attack_index = random.randint(0, len(self.attacks) - 1)
            attack_name, attack_for_batch = self.attacks[attack_index]

            batch_x, mn, mx = utils.to_minmax(batch_x)
            batch_x = attack_for_batch(batch_x, batch_y)
            batch_x = utils.revert_minmax(batch_x, mn, mx)

        return batch_x

    def update_adv_attack(self, batch_loss, batch_pred, iter=None, epoch=None):
        ...


class EqualAdversarialGDTrainer(AdversarialGDTrainer):
    def apply_adv_attack(self, batch_x, batch_y,):
        attack_index = 0
        # Get indices of samples to attack
        batch_indices = range(len(batch_x))
        attack_name, attack_for_batch = self.attacks[attack_index]

        # Get batch_size // 2 random samples, extract them, attack and replace in original batch
        indices_to_attack = random.sample(batch_indices, len(batch_x) // 2)
        attacked_samples = batch_x[indices_to_attack]
        attacked_ys = batch_y[indices_to_attack]

        attacked_samples, mn, mx = utils.to_minmax(attacked_samples)
        attacked_samples = attack_for_batch(attacked_samples, attacked_ys)
        attacked_samples = utils.revert_minmax(attacked_samples, mn, mx)

        batch_x[indices_to_attack, ...] = attacked_samples

        return batch_x


class OnlyOneAdversarialGDTrainer(AdversarialGDTrainer):

    def init_adv_attacks(self, attack_model, adversarial_attacks):
        assert len(adversarial_attacks) == 1, "Method allows to apply only one attack"
        self.attacks = super().init_adv_attacks(attack_model, adversarial_attacks)
        return self.attacks

    def apply_adv_attack(self, batch_x, batch_y):
        attack_name, attack_for_batch = self.attacks[0]

        batch_x, mn, mx = utils.to_minmax(batch_x)
        batch_x = attack_for_batch(batch_x, batch_y)
        batch_x = utils.revert_minmax(batch_x, mn, mx)

        return batch_x


class AdaptiveAdversarialGDTrainer(AdversarialGDTrainer):

    def __init__(self, *args, **kwargs):
        super(AdaptiveAdversarialGDTrainer, self).__init__(*args, **kwargs)

        self.adv_attacks_weights = None
        self.last_adv_attack = None

    def init_adv_attacks(self, attack_model, adversarial_attacks):
        self.attacks = super().init_adv_attacks(attack_model, adversarial_attacks)
        self.adv_attacks_weights = [1/(len(self.attacks)+1)] * (len(self.attacks)+1)

        return self.attacks

    def apply_adv_attack(self, batch_x, batch_y):
        attack_idx, = random.choices(range(len(self.attacks)+1), weights=self.adv_attacks_weights, k=1)
        self.last_adv_attack = attack_idx

        if attack_idx < len(self.attacks):
            attack_name, attack_for_batch = self.attacks[attack_idx]

            batch_x, mn, mx = utils.to_minmax(batch_x)
            batch_x = attack_for_batch(batch_x, batch_y)
            batch_x = utils.revert_minmax(batch_x, mn, mx)

        return batch_x

    def update_adv_attack(self, batch_loss, batch_pred, max_val=1, proportion_val=0.2, iter=None, epoch=None):
        loss = min(batch_loss, max_val)

        self.adv_attacks_weights[self.last_adv_attack] = \
            proportion_val * loss + (1-proportion_val) * self.adv_attacks_weights[self.last_adv_attack]

        weights_sum = np.sum(self.adv_attacks_weights)
        self.adv_attacks_weights = [
            0.5 * (w / weights_sum) + 0.5 * (1.0 / len(self.adv_attacks_weights))
            for w in self.adv_attacks_weights
        ]

        if iter is not None and iter % 100 == 0:
            LOGGER.info(f"[{epoch:04d}][{iter:05d}]: Adversarial attack weights: {self.adv_attacks_weights}")


class AdaptiveV2AdversarialGDTrainer(AdaptiveAdversarialGDTrainer):
    def update_adv_attack(self, batch_loss, batch_pred, max_val=1, proportion_val=0.2, iter=None, epoch=None):
        loss = min(batch_loss, max_val)

        self.adv_attacks_weights[self.last_adv_attack] = \
            proportion_val * loss + (1-proportion_val) * self.adv_attacks_weights[self.last_adv_attack]

        weights_sum = np.sum(self.adv_attacks_weights)
        self.adv_attacks_weights = [0.5 * (w / weights_sum) for w in self.adv_attacks_weights]

        non_attack_ratio = 1/3
        attack_ratio = (2/3) / len(self.attacks)

        self.adv_attacks_weights = [
            w + 0.5 * attack_ratio
            if i < len(self.adv_attacks_weights)-1
            else w + 0.5 * non_attack_ratio
            for i, w in enumerate(self.adv_attacks_weights)
        ]

        if iter is not None and iter % 100 == 0:
            LOGGER.info(f"[{epoch:04d}][{iter:05d}]: Adversarial attack weights: {self.adv_attacks_weights}")
