from enum import Enum
from adversarial_attacks import torchattacks


class AttackEnum(Enum):

    # PGD is implemented while not used in the paper
    PGD = (torchattacks.PGD, {"eps": 0.0005, "steps": 10})
    PGD_eps00075 = (torchattacks.PGD, {"eps": 0.00075, "steps": 10})
    PGD_eps001 = (torchattacks.PGD, {"eps": 0.001, "steps": 10})

    PGDL2 = (torchattacks.PGDL2, {"eps": 0.1,  "steps": 10})
    PGDL2_eps15 = (torchattacks.PGDL2, {"eps": 0.15,  "steps": 10})
    PGDL2_eps20 = (torchattacks.PGDL2, {"eps": 0.20,  "steps": 10})

    FGSM = (torchattacks.FGSM, {"eps": 0.0005})
    FGSM_eps00075 = (torchattacks.FGSM, {"eps": 0.00075})
    FGSM_eps001 = (torchattacks.FGSM, {"eps": 0.001})

    FAB = (torchattacks.FAB, {"n_classes": 2, "eta": 10})
    FAB_eta20 = (torchattacks.FAB, {"n_classes": 2, "eta": 20})
    FAB_eta30 = (torchattacks.FAB, {"n_classes": 2, "eta": 30})

    NO_ATTACK = (None, {})


