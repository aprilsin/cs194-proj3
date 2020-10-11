# morphing sequence

import math
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage as sk
import skimage.io as io
from skimage import util
from skimage.util import img_as_float, img_as_ubyte

import utils

NUM_FRAMES = 10

def get_affine_mat(
    start: list, target: list
) -> np.ndarray:
    
    # A*T = B
    # T = A-1 * B
    # return invA * B
    A = np.array([start[:,0], start[:, 1], [1, 1, 1]])
    B = np.array([target[:,0], target[:, 1], [1, 1, 1]])
    return np.linalg.inv(A) * B


def cross_dissolve():
    pass


def morph():
    # for each timeframe
    pass