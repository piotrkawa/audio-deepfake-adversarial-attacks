from enum import Enum

from src.trainer import (
    AdaptiveAdversarialGDTrainer,
    AdaptiveV2AdversarialGDTrainer,
    AdversarialGDTrainer,
    EqualAdversarialGDTrainer,
    OnlyOneAdversarialGDTrainer
)


class AdversarialGDTrainerEnum(Enum):
    ONLY_ADV = OnlyOneAdversarialGDTrainer
    RANDOM = AdversarialGDTrainer
    ADAPTIVE = AdaptiveAdversarialGDTrainer
    ADAPTIVE_V2 = AdaptiveV2AdversarialGDTrainer
    EQUAL = EqualAdversarialGDTrainer
