from enum import Enum

from src.trainer import AdaptiveV2AdversarialGDTrainer, AdversarialGDTrainer, AdaptiveAdversarialGDTrainer, OnlyOneAdversarialGDTrainer, EqualAdversarialGDTrainer


class AdversarialGDTrainerEnum(Enum):
    ONLY_ADV = OnlyOneAdversarialGDTrainer
    RANDOM = AdversarialGDTrainer
    ADAPTIVE = AdaptiveAdversarialGDTrainer
    ADAPTIVE_V2 = AdaptiveV2AdversarialGDTrainer
    EQUAL = EqualAdversarialGDTrainer

