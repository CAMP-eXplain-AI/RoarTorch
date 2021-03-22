import random

import numpy as np
import torch


def set_random_seed(random_seed: int):
    """
    Set seed for random seed generator for python, pytorch and numpy.
    :param random_seed: Initial seed value
    """
    print(f'Using Random Seed value as: {random_seed}')
    torch.manual_seed(random_seed)  # Set for pytorch, used for cuda as well.
    random.seed(random_seed)  # Set for python
    np.random.seed(random_seed)  # Set for numpy
