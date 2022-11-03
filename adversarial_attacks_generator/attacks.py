from enum import Enum

from adversarial_attacks import torchattacks


class AttackEnum(Enum):

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

    CW = (torchattacks.CW, {"c": 1e-4})
    CW_c3 = (torchattacks.CW, {"c": 1e-3})
    CW_c2 = (torchattacks.CW, {"c": 1e-2})

    APGDT = (torchattacks.APGDT, {"norm": 'L2', "eps": 0.1, "steps": 100})    # very weak

    ONE_PIXEL = (torchattacks.OnePixel, {"pixels": 20})     # does not work well
    NO_ATTACK = (None, {})
