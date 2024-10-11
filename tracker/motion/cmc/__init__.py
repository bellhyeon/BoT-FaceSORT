from tracker.motion.cmc.ecc import ECC
from tracker.motion.cmc.orb import ORB
from tracker.motion.cmc.sift import SIFT
from tracker.motion.cmc.sof import SparseOptFlow


def get_cmc_method(cmc_method):
    if cmc_method == "ecc":
        return ECC
    elif cmc_method == "orb":
        return ORB
    elif cmc_method == "sof":
        return SparseOptFlow
    elif cmc_method == "sift":
        return SIFT
    else:
        return None
