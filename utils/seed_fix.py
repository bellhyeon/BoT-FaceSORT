import random
import os
import torch
import numpy as np


def fix_seed(seed: int = 42):
    """
    fix seed to control any randomness from a code
    (enable stability of the experiments' results.)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
